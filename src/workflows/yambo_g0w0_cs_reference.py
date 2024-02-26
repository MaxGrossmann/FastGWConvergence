"""
This workflows performs reference GW calculation 
with very high parameters in W on multiple k-point grids.
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
wf_dir = os.path.join(os.getcwd(), "yambo_g0w0_cs_reference")
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

# parameters for every gw reference calculation
bnd_max = 1200
cut_max = 46

# main loop over different k-point grids
kppa_grid = [0, 10, 50, 100, 500]

# number of k-point grids
nk = len(kppa_grid)

# save all the k-point grids
k_grid = np.zeros([nk + 1, 3], dtype=int)

# save the gap to check for the k-point grid convergence
k_gap = np.zeros(nk + 1)

# cutoff
pw_cutoff = None

# create the output files
with open(f"{id}_gw_conv_cs_ref.txt", "w+") as f:
    f.write(
        f"kppa \t kpt \t    nk_ir  bnd_x  bnd_g  cutsrc  ref_gap   ref_time (s)  ncpu\n"
    )

# loop over different k-point grids
for k in range(nk):
    # start the time
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
                kppa_grid[k] += 10
                k_points_grid = get_kpt_grid(structure, kppa_grid[k])
                print(
                    f"Updating k-point grid density to kppa = {kppa_grid[k]:d}...", flush=True
                )

        # update the k-point grid array and increase the next k-point density accordingly
        k_grid[k + 1, :] = k_points_grid
        if k < nk - 1:
            kppa_grid[k + 1] = kppa_grid[k] + 10

    # for write out later on
    print(
        f"\nUsing the following k-point grid from density kppa = {kppa_grid[k]:d}:",
        flush=True,
    )
    print(k_grid[k + 1, :], flush=True)
    print("", flush=True)

    # run the reference calculation
    (
        ref_gap,
        num_kpt,
        filename_scf,
        filename_nscf,
        pw_cutoff,
    ) = yambo_runner.yambo_run_gw_conv_cs_reference(
        calc_data_conv,
        kppa_grid[k],
        base_dir,
        ncpu,
        bnd_max=bnd_max,
        cut_max=cut_max,
        pw_cutoff=pw_cutoff,
    )

    # append the output file
    kstr = f"{k_grid[k+1, 0]:d}x{k_grid[k+1, 1]:d}x{k_grid[k+1, 2]:d}"
    with open(f"{id}_gw_conv_cs_ref.txt", "a+") as f:
        f.write(f"{kppa_grid[k]:<4d} \t {kstr:<9}  {num_kpt:<5d}  ")
        f.write(
            f"{int(bnd_max):<5d}  {int(bnd_max):<5d}  {int(cut_max):<6d}  {ref_gap:<2.5f}  "
        )
        f.write(f"{int(np.ceil((time.time() - start_time))):<12d}  {ncpu:<4d}\n")

# delete the qe output folder
shutil.rmtree("out")
