"""
This workflow finds the convergence parameters for a GW calculation
in a very efficient way. First, the convergence parameters for W 
are found using the coordinate search algorithm on a Gamma-only k-point
grid. Then, the k-point grid density is increased and one GW calculation
per k-point grid is performed using small parameters in W, i.e. (200,4), 
until the gap converges with respect to the k-point grid.
In some cases, this may result in an underconverged calculation.
If high accuracy is required, check the final parameters again.
For high-throughput calculations, this workflow is safe.
"""

# external imports
import os
import sys
import time
import shutil
import numpy as np

# local imports
from src.utils.calc_data_class import calc_data
import src.utils.qe_write as qe_write
import src.utils.qe_helper as qe_helper
import src.utils.qe_runner as qe_runner
import src.utils.yambo_runner as yambo_runner
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write
from src.utils.basic_utils import get_kpt_grid

# base directory
base_dir = str(sys.argv[1])

# number of cores
ncores = int(sys.argv[2])

# get the id from the args that are called with this script
id = str(sys.argv[3])

# workflow directory
wf_dir = os.path.join(os.getcwd(), "yambo_g0w0_conv")
if os.path.exists(wf_dir):
    shutil.rmtree(wf_dir)
if not os.path.exists(wf_dir):
    os.mkdir(wf_dir)
os.chdir(wf_dir)

# initialize structure for a QuantumEspresso calculation
structure, name, ibrav = qe_helper.qe_init_structure(id, base_dir)
vol = qe_helper.uc_vol_au(structure)

# flag for 2d materials
flag_2d = False
aspect_ratio = max(structure.lattice.abc) / min(structure.lattice.abc)
if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
    print("\n2D MATERIAL", flush=True)
    flag_2d = True

# setup and start a scf calculation on a fine k-point grid
convergence_flag, calc_data_conv = qe_runner.qe_convergence_checker(
    id, wf_dir, ncores, conv_dir="qe_convergence_SG15"
)

# number of electrons
num_elec = qe_helper.qe_get_electrons(calc_data_conv)

# use the npj input style?
# this was just a test, to make sure that the yambo
# input style does not affect the calculations
write_npj = False

# good starting value for the maximum number of bands in the nscf
# this will be increased if the convergence fails
n_bands = 800

# convergence threshold (if conv_percent > 0 it is used instead of conv_thr...)
conv_thr = 0.025  # eV

# starting parameter and parameter step size
bnd_start = 200
bnd_step = 100
cut_start = 4
cut_step = 4

# calculate the reference gap with maximum convergence parameters
# (see yambo_g0w0_cs_reference.py workflow)
ref_flag = False

# start the time
start_time = time.time()

# change the number of cores for small k-point grid calculations
# as the parallel structure will not work well with too many cores
# and too few k-points (same applies to nscf calculations)
if num_elec < 5:
    ncpu = np.min([4, ncores])
else:
    ncpu = np.min([8, ncores])

# create the output files
with open(f"{id}_gw_conv.txt", "w+") as f:
    f.write(
        f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  gap      #GW  time (s)  ncpu\n"
    )

# converge the parameters in W doing a gamma-only gw calculation
gw = None
n_gw_calc = 0
while True:
    (
        conv_flag,
        bnd_increase_flag,
        gw,
        filename_scf,
        filename_nscf,
        pw_cutoff,
    ) = yambo_runner.yambo_run_gw_conv_cs(
        calc_data_conv,
        n_bands,
        0,
        gw,
        base_dir,
        ncpu,
        bnd_start=bnd_start,
        bnd_step=bnd_step,
        cut_start=cut_start,
        cut_step=cut_step,
        conv_thr=conv_thr,
        ref_flag=ref_flag,
    )
    n_gw_calc += len(gw.fn)
    if conv_flag:
        print("CONVERGED.\n", flush=True)
        break
    elif not conv_flag and bnd_increase_flag:
        print("Increasing number of bands in the nscf step...\n", flush=True)
        os.remove(f"{filename_scf}.in")
        os.remove(f"{filename_scf}.out")
        os.remove(f"{filename_nscf}.in")
        os.remove(f"{filename_nscf}.out")
        shutil.rmtree("out")
        n_bands += int(2.5 * bnd_step)
        if n_bands >= 3000:
            print("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED.\n", flush=True)
            break
    elif not conv_flag and not bnd_increase_flag:
        print("CONVERGENCE FAILED. MAXIMUM CUTOFF WAS REACHED.", flush=True)

# append the output file
kstr = f"{1:d}x{1:d}x{1:d}"
with open(f"{id}_gw_conv.txt", "a+") as f:
    if n_bands >= 3000:
        f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED\n")
    elif not conv_flag and not bnd_increase_flag:
        f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED.\n")
    else:
        f.write(f"{0:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
        f.write(
            f"{int(gw.final_point[0]):<5d}  {int(gw.final_point[0]):<5d}  {int(gw.final_point[1]):<6d}  "
            + f"{gw.final_point[2]:<2.5f}  {n_gw_calc:<3d}  "
        )
    f.write(f"{int(np.ceil((time.time() - start_time))):<8d}  {ncpu:<4d}\n")

    # write output the results from the starting point as a reference for the k-point grid convergence
    f.write(f"{0:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
    f.write(
        f"{int(gw.grid[0][0]):<5d}  {int(gw.grid[0][0]):<5d}  {int(gw.grid[0][1]):<6d}  "
        + f"{gw.grid[0][2]:<2.5f}  {1:<3d}  "
    )
    f.write(f"{int(np.ceil(gw.grid_time[0])):<8d}  {ncpu:<4d}\n")

# now converge the k-point grid using the cheap starting values from W
kppa = [0]
k_grid = [np.array(get_kpt_grid(structure, kppa[0]))]
kgrid_gap = [gw.grid[0][2]]
n_bands = gw.grid[0][0]
diff_gap = 1
iter = 1
max_iter = 7
while diff_gap > conv_thr:
    # start the time
    start_time = time.time()

    # increase the k-point grid density
    kppa.append(kppa[-1] + 10)

    # change the number of cores for small k-point grid calculations
    # as the parallel structure is not working well with too many cores
    # (also the nscf has convergence problems)
    if kppa[iter] < 50:
        if num_elec < 5:
            ncpu = np.min([4, ncores])
        else:
            ncpu = np.min([8, ncores])
    else:
        ncpu = ncores

    # obtain the new k-point grid for the nscf step
    k_grid.append(np.array(get_kpt_grid(structure, kppa[iter])))

    # increase the k-point density until the k-point grid changes
    while np.sum(np.abs(k_grid[iter] - k_grid[iter - 1])) == 0:
        kppa[iter] += 10
        k_grid[iter] = np.array(get_kpt_grid(structure, kppa[iter]))
        print(f"Updating k-point grid density to kppa = {kppa[iter]:d}...", flush=True)

    # for write out later on
    print(
        f"\nUsing the following k-point grid from density kppa = {kppa[iter]:d}:",
        flush=True,
    )
    print(k_grid[iter], flush=True)
    print("", flush=True)

    # create nscf calculation based on scf calculation and run it
    calc_data_nscf = calc_data(
        structure,
        name,
        id=id,
        ibrav=ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa[iter],
        pseudo=os.path.join(base_dir, "pseudo", "SG15"),
    )

    # reuse the nscf settings from the W convergence
    filename_nscf = qe_runner.qe_pw_run(
        calc_data_nscf,
        qe_write.write_nscf_yambo,
        ncpu,
        kwargs={"n_bands": n_bands},
    )

    # create a yambo subfolder
    if not os.path.exists(f"kppa{kppa[iter]:d}"):
        os.mkdir(f"kppa{kppa[iter]:d}")

    # p2y step with the output redirected to the yambo folder
    os.chdir(f"out/{id}.save")
    os.system(f"p2y -O ../../kppa{kppa[iter]:d}/")

    # move to the subfolder for the yambo calculation
    os.chdir(f"../../kppa{kppa[iter]:d}")
    print(f"Current directory: kppa{kppa[iter]:d}", flush=True)

    # yambo setup
    os.system("yambo")

    # get the number of electrons in the unit cell
    num_elec = yambo_helper.get_num_electrons("r_setup")
    if num_elec & 0x1:
        raise Exception("Uneven number of electrons in the unit cell!")

    # get the total number of k-points
    num_kpt = yambo_helper.get_num_kpt("r_setup")

    # read the r_setup to find where the direct gap is situated
    kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters("r_setup")
    if kpt_bnd_idx[2] < num_elec / 2:
        print("Metallic states are present...", flush=True)
        kpt_bnd_idx[2] = int(num_elec / 2)
        kpt_bnd_idx[3] = int(num_elec / 2 + 1)

    # test our coordinate search algorithm
    if not os.path.isdir("g0w0"):
        os.mkdir("g0w0")
    os.chdir("g0w0")

    # create the input file and start the calculation
    f_name = yambo_write.write_g0w0(
        bnd_start,
        cut_start,
        bnd_start,
        kpt_bnd_idx,
        flag_2d=flag_2d,
    )
    os.system(f"mpirun -np {ncpu} yambo -F {f_name}.in -J {f_name} -I ../")
    kgrid_gap.append(yambo_helper.get_minimal_gw_gap(f_name, kpt_bnd_idx))

    # go back up to the main directory
    os.chdir("../../")

    # append the output file
    kstr = f"{k_grid[iter][0]:d}x{k_grid[iter][1]:d}x{k_grid[iter][2]:d}"
    with open(f"{id}_gw_conv.txt", "a+") as f:
        f.write(f"{kppa[iter]:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
        f.write(
            f"{bnd_start:<5d}  {bnd_start:<5d}  {cut_start:<6d}  "
            + f"{kgrid_gap[iter]:<2.5f}  {1:<3d}  "
        )
        f.write(f"{int(np.ceil((time.time() - start_time))):<8d}  {ncpu:<4d}\n")

    # calculate the difference
    diff_gap = np.abs(kgrid_gap[iter] - kgrid_gap[iter - 1])
    print(f"Gap difference: {diff_gap:.6f}\n", flush=True)

    # convergence condition
    if diff_gap < conv_thr:
        print("CONVERGED.\n", flush=True)
        break

    # break condition
    if iter + 1 > max_iter:
        print("k-point grid DID NOT CONVERGE.\n", flush=True)
        break

    # log message
    print(
        f"Finshed the kppa = {kppa[iter]:d} workflow, going back to {os.getcwd()}\n",
        flush=True,
    )
    iter += 1

# write output file with the final parameters
with open("gw_conv_params.txt", "w") as f:
    f.write("bands  cutoff  kppa\n")
    f.write(f"{int(gw.final_point[0]):<5d}  {int(gw.final_point[1]):<6d}  {kppa[-1]:d}")

# message for the log file
print("DETERMINATION OF THE GW CONVERGENCE PARAMETERS FINISHED!\n", flush=True)
