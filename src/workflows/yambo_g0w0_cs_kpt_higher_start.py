"""
This workflows performs finds the convergence parameters
for W using the coordinate search algorithm on multiple k-point grids.
This version uses a higher starting point for the CS algorithm to analyze
the starting point dependence of the convergence.
"""

# external imports
import os
import sys
import time
import shutil
import numpy as np

# local imports
import src.utils.qe_helper as qe_helper
import src.utils.qe_runner as qe_runner
import src.utils.yambo_runner as yambo_runner
from src.utils.basic_utils import get_kpt_grid

# base directory
base_dir = str(sys.argv[1])

# number of cores
ncores = int(sys.argv[2])

# get the id from the args that are called with this script
id = str(sys.argv[3])

# workflow directory
wf_dir = os.path.join(os.getcwd(), "yambo_g0w0_cs_kpt_higher_start")
if os.path.exists(wf_dir):
    shutil.rmtree(wf_dir)
if not os.path.exists(wf_dir):
    os.mkdir(wf_dir)
os.chdir(wf_dir)

# initialize structure for a QuantumEspresso calculation
structure, name, ibrav = qe_helper.qe_init_structure(id, base_dir)
vol = qe_helper.uc_vol_au(structure)

# setup and start a scf calculation on a fine k-point grid
convergence_flag, calc_data_conv = qe_runner.qe_convergence_checker(
    id, wf_dir, ncores, conv_dir="qe_convergence_SG15"
)

# number of electrons
num_elec = qe_helper.qe_get_electrons(calc_data_conv)

# use the npj input style?
write_npj = False

# good starting value for the maximum number of bands in the nscf
# this will be increased if the convergence fails
n_bands = 1200

# convergence threshold (if conv_percent > 0 it is used instead of conv_thr...)
conv_thr = 0.025  # eV
aspect_ratio = max(structure.lattice.abc) / min(structure.lattice.abc)
if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
    conv_thr = 0.05  # eV (higher threshold for 2D systems)

# starting parameter and parameter step size
bnd_start = 400
bnd_step = 100
cut_start = 12
cut_step = 4

# calculate the reference gap with maximum convergence parameters
ref_flag = False

# main loop over different k-point grids
aspect_ratio = max(structure.lattice.abc) / min(structure.lattice.abc)
if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
    print("\n2D MATERIAL", flush=True)
    kppa_grid = [0, 1, 2, 3, 4, 5, 6, 7]
else:  
    kppa_grid = [0, 10, 50, 100, 500]

# number of k-point grids
nk = len(kppa_grid)

# save all the k-point grid
k_grid = np.zeros([nk + 1, 3], dtype=int)

# save the gap to check for the k-point grid convergence
k_gap = np.zeros(nk + 1)

# create the output files
with open(f"{id}_gw_conv_cs.txt", "w+") as f:
    if ref_flag:
        f.write(
            f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  gap      #GW  time (s)  ncpu  ref_gap  ref_time\n"
        )
    else:
        f.write(
            f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  gap      #GW  time (s)  ncpu\n"
        )

# loop over different k-point grids
for k in range(nk):
    # start time
    start_time = time.time()

    # change the number of cores for small k-point grid calculations
    # as the parallel structure is not working well with too many cores
    # (also the nscf has convergence problems)
    if kppa_grid[k] < 50:
        if num_elec < 5:
            ncpu = np.min([4, ncores])
        else:
            ncpu = np.min([8, ncores])
    else:
        ncpu = ncores

    # obtain the k-point grid for the nscf step
    k_points_grid = get_kpt_grid(structure, kppa_grid[k])

    # add the new grid
    k_grid[k + 1, :] = k_points_grid

    # increase the k-point density until the k-point grid changes
    if len(kppa_grid) > 1:
        if np.sum(np.abs(k_grid[k + 1, :] - k_grid[k, :])) == 0:
            while np.sum(np.abs(k_grid[k + 1] - k_points_grid)) == 0:
                if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
                    kppa_grid[k] += 1
                else:
                    kppa_grid[k] += 10
                k_points_grid = get_kpt_grid(structure, kppa_grid[k])
                print(
                    f"Updating k-point grid density to kppa = {kppa_grid[k]:d}...", flush=True
                )

        # update the k-point grid array and increase the next k-point density accordingly
        k_grid[k + 1, :] = k_points_grid
        if k < nk - 1:
            if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
                kppa_grid[k + 1] = kppa_grid[k] + 1
            else:
                kppa_grid[k + 1] = kppa_grid[k] + 10

    # for write out later on
    print(
        f"\nUsing the following k-point grid from density kppa = {kppa_grid[k]:d}:",
        flush=True,
    )
    print(k_grid[k + 1, :], flush=True)
    print("", flush=True)

    # run the convergence, if it fails increase the number of bands and restart
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
            kppa_grid[k],
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
            n_bands += int(2.5 * bnd_step)
            if n_bands >= 3000:
                print(
                    "CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED.\n", flush=True
                )
                break
        elif not conv_flag and not bnd_increase_flag:
            print("CONVERGENCE FAILED. MAXIMUM CUTOFF WAS REACHED.", flush=True)
            break

    # append the output file
    kstr = f"{k_grid[k+1, 0]:d}x{k_grid[k+1, 1]:d}x{k_grid[k+1, 2]:d}"
    with open(f"{id}_gw_conv_cs.txt", "a+") as f:
        if n_bands >= 3000:
            f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED\n")
        elif not conv_flag and not bnd_increase_flag:
            f.write("CONVERGENCE FAILED. TOO MANY BANDS WERE REQUESTED.\n")
        else:
            f.write(f"{kppa_grid[k]:<4d} \t {kstr:<9}  {gw.num_kpt:<5d}  ")
            f.write(
                f"{int(gw.final_point[0]):<5d}  {int(gw.final_point[0]):<5d}  {int(gw.final_point[1]):<6d}  "
                + f"{gw.final_point[2]:<2.5f}  {n_gw_calc:<3d}  "
            )
            if ref_flag:
                f.write(
                    f"{int(np.ceil((time.time() - start_time))):<8d}  {ncpu:<4d}  {gw.ref_gap:<2.5f}  {int(np.ceil(gw.ref_time)):<7d}\n"
                )
            else:
                f.write(f"{int(np.ceil((time.time() - start_time))):<8d}  {ncpu:<4d}\n")

# delete the qe output folder
shutil.rmtree("out")
