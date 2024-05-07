"""
This workflow runs a simple convergence algorithm to converge 
the plane-wave cutoff and the number of k-points for a given material. 
The convergence is in regards to the total energy per atom in the unit cell.
"""

# external imports
import os
import sys
import time
import pickle
import numpy as np

# local imports
import src.utils.basic_utils as basic_utils
import src.utils.qe_helper as qe_helper
import src.utils.qe_write as qe_write
import src.utils.qe_runner as qe_runner
from src.utils.calc_data_class import calc_data

# start timing
start_time = time.time()

# base directory
base_dir = str(sys.argv[1])

# number of cores
ncores = int(sys.argv[2])

# get the id from the args that are called with this script
id = str(sys.argv[3])

# workflow directory
wf_dir = os.path.join(os.getcwd(), "qe_convergence_SG15")
if not os.path.exists(wf_dir):
    os.mkdir(wf_dir)
os.chdir(wf_dir)

# do you want to delete the wavefunctions?
# right now all the workflows, if necessary, generate their own WFs
delete_wfc = True

# some sensible defaults for high-throughput calculation
maxiter = 10
conv_thresh = 0.04  # eV, set for "chemical accuracy", e.g. 1kcal/mol

# step sizes for the convergence algorithm
delta_kppa = 1500
delta_cutoff = 5

# initialize structure for a QuantumEspresso calculation
structure, name, ibrav = qe_helper.qe_init_structure(id, base_dir)

# log message if the material is 2D
aspect_ratio = max(structure.lattice.abc) / min(structure.lattice.abc)
if structure.lattice.abc[2] > 15 and aspect_ratio > 5:
    print("\n2D MATERIAL", flush=True)

# create dummy scf calculation (useful to check k-grid later)
calc_data0 = calc_data(
    structure,
    name,
    id=id,
    ibrav=ibrav,
    calc_type="scf",
    kppa=1500,
    pseudo=os.path.join(base_dir, "pseudo", "SG15"),
)

# extract the initial parameters and create a calculation which is changed later on
structure = calc_data0.structure
name = calc_data0.name
id = calc_data0.id
ibrav = calc_data0.ibrav
pw_cutoff = calc_data0.pw_cutoff
kppa = calc_data0.kppa
pseudo = calc_data0.pseudo
calc_data_curr = calc_data(
    structure,
    name,
    id=id,
    ibrav=ibrav,
    pw_cutoff=pw_cutoff,
    kppa=kppa,
    pseudo=pseudo,
)

# convert the convergence threshold to Ha
# we also converge the energy per atom in the unit cell
conv_thresh = basic_utils.ev2ha(conv_thresh)
conv_thresh = conv_thresh * len(structure.sites)

# collect file names
fn = []

# list of energies after each step
energies = []

# list of convergence parameters
conv_param = [[pw_cutoff, kppa]]

# maximum number of calculations
iter = 0

# obtain starting values
filename = qe_runner.qe_pw_run(
    calc_data_curr, qe_write.write_scf, ncores, kwargs={"disk_io": "low"}
)
fn.append(filename)

# k-point convergence

# read out energy from output file, print it and add it to the energies list
curr_energy = qe_helper.qe_read_totenergy(id)
energies.append(curr_energy)

# create a file to document the results of the convergence calculations
with open("conv_results.txt", "w+") as f:
    f.write(f"convergence threshold = {conv_thresh:.4f} Ha (= 1kcal/mol)\n")
    f.write("cutoff (Ry)  kppa   total energy (Ha)\n")
    f.write(f"{conv_param[-1][0]:<11d}  {conv_param[-1][1]:<5d}  {energies[-1]:.6f}\n")

# increase k-point density, run calculation again etc.
calc_data0 = calc_data_curr

# check whether k_point_grid is actually refined
while calc_data_curr.k_points_grid == calc_data0.k_points_grid:
    calc_data0 = calc_data_curr
    kppa += delta_kppa
    calc_data_curr = calc_data(
        structure,
        name,
        id=id,
        ibrav=ibrav,
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=pseudo,
    )

# start calculation with a finer k-point grid
filename = qe_runner.qe_pw_run(
    calc_data_curr, qe_write.write_scf, ncores, kwargs={"disk_io": "low"}
)
fn.append(filename)
iter += 1

# read out energy from output file, print it and add it to the energies list
curr_energy = qe_helper.qe_read_totenergy(id)
energies.append(curr_energy)

# save the convergence parameters
conv_param.append([pw_cutoff, kppa])

# append results of the convergence calculations
with open("conv_results.txt", "a+") as f:
    f.write(f"{conv_param[-1][0]:<11d}  {conv_param[-1][1]:<5d}  {energies[-1]:.6f}\n")

# compare energies between coarse and finer k-point grid
# while energie difference is above threshold, make k-point grid finer, repeat calculation etc.
while abs(energies[-1] - energies[-2]) > conv_thresh:
    calc_data0 = calc_data_curr
    # check whether k_point_grid is actually refined
    while calc_data_curr.k_points_grid == calc_data0.k_points_grid:
        calc_data0 = calc_data_curr
        kppa += delta_kppa
        calc_data_curr = calc_data(
            structure,
            name,
            id=id,
            ibrav=ibrav,
            pw_cutoff=pw_cutoff,
            kppa=kppa,
            pseudo=pseudo,
        )
    # start calculation with finer k-point grid
    filename = qe_runner.qe_pw_run(
        calc_data_curr, qe_write.write_scf, ncores, kwargs={"disk_io": "low"}
    )
    fn.append(filename)
    iter += 1

    # read out energy from output file, print it and add it to the energies list
    curr_energy = qe_helper.qe_read_totenergy(id)
    energies.append(curr_energy)
    conv_param.append([pw_cutoff, kppa])

    # append results of the convergence calculations
    with open("conv_results.txt", "a+") as f:
        f.write(
            f"{conv_param[-1][0]:<11d}  {conv_param[-1][1]:<5d}  {energies[-1]:.6f}\n"
        )

    # safety feature
    if iter >= maxiter:
        print(
            f"K-point density not converged after {maxiter} iterations. Proceeding to Cutoff convergence."
        )
        break

# cutoff convergence

# the same as above, but this time for the cutoff energy
iter = 0
pw_cutoff += delta_cutoff
calc_data_curr = calc_data(
    structure,
    name,
    id=id,
    ibrav=ibrav,
    pw_cutoff=pw_cutoff,
    kppa=kppa,
    pseudo=pseudo,
)

# start calculation with higher cutoff energy
filename = qe_runner.qe_pw_run(
    calc_data_curr, qe_write.write_scf, ncores, kwargs={"disk_io": "low"}
)
fn.append(filename)
iter += 1

# read out energy from output file, print it and add it to the energies list
curr_energy = qe_helper.qe_read_totenergy(id)
energies.append(curr_energy)
conv_param.append([pw_cutoff, kppa])

# append results of the convergence calculations
with open("conv_results.txt", "a+") as f:
    f.write(f"{conv_param[-1][0]:<11d}  {conv_param[-1][1]:<5d}  {energies[-1]:.6f}\n")

while abs(energies[-1] - energies[-2]) > conv_thresh:
    pw_cutoff += delta_cutoff
    calc_data_curr = calc_data(
        structure,
        name,
        id=id,
        ibrav=ibrav,
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=pseudo,
    )

    # start calculation with higher cutoff energy
    filename = qe_runner.qe_pw_run(
        calc_data_curr, qe_write.write_scf, ncores, kwargs={"disk_io": "low"}
    )
    fn.append(filename)
    iter += 1

    # read out energy from output file, print it and add it to the energies list
    curr_energy = qe_helper.qe_read_totenergy(id)
    energies.append(curr_energy)
    conv_param.append([pw_cutoff, kppa])

    # append results of the convergence calculations
    with open("conv_results.txt", "a+") as f:
        f.write(
            f"{conv_param[-1][0]:<11d}  {conv_param[-1][1]:<5d}  {energies[-1]:.6f}\n"
        )

    # safety feature
    if iter >= maxiter:
        print(f"Cutoff not converged after {maxiter} iterations.")
        break

# sort the convergence parameters
unique_cutoff = np.array(conv_param)[:, 0]
unique_cutoff = np.unique(unique_cutoff)
unique_kppa = np.array(conv_param)[:, 1]
unique_kppa = np.unique(unique_kppa)
unique_cutoff.sort()
unique_kppa.sort()
calc_data_curr.pw_cutoff = unique_cutoff[-1]
calc_data_curr.kppa = unique_kppa[-1]

# save the final calculation data class to a file
f = open("conv_calc_data.pckl", "wb")
pickle.dump(calc_data_curr, f)
f.close()

# delete unwanted files
for f in fn[:-1]:
    os.remove(f + ".in")
    os.remove(f + ".out")

# delete wavefunctions (if wanted)
if delete_wfc:
    for filepath in os.listdir(os.path.join(wf_dir, "out", id + ".save")):
        if "wfc" in filepath:
            os.remove(os.path.join(wf_dir, "out", id + ".save", filepath))

# save calculation time
with open("timing.txt", "a+") as f:
    f.write(
        f"{os.path.basename(__file__):<25}  {(time.time() - start_time):7.2f} s  {ncores} cores\n"
    )
