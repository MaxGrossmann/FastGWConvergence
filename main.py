# external imports
import os
import pickle

# local imports
from api_key import *
import src.utils.basic_utils as basic_utils

"""
START OF USER INPUT SECTION
"""

# calculation setup ("local" or "batchjob")
# (local is mostly just used for testing and debugging purposes)
# (for batchjob you need to adjust the "start_calc" function in src/utitls/basic_utils.py
#  to suit your local computing environment, which depends on our job submission system.)
calc_setup = "local"

# number of cores per job
ncores = 16

# job memory (does nothing when doing local alculation)
# (this could be estimated in the future depending on the system size and other parameters)
memory = 65536  # MB

# calculation type defined through script names (see "/src/workflows")
# arrays of workflows that will be done one after each other
script_name = [
    "qe_convergence_SG15",
    "yambo_g0w0_cs_kpt",
    "yambo_g0w0_npj_kpt",
    "yambo_g0w0_cs_reference",
]

# material IDs from the Materials Project
# (for local calculation only use one material at a time!)
material_id = [
    "mp-66",
]

"""
END OF USER INPUT SECTION
"""

# name of the calculation folder
# (this needs to be changed to "ref" when doing the npj/sota reference calculations)
calc_folder = "calc"

# get the base directory so each script knows where the source folder is
base_dir = os.getcwd()

# loop over all materials
for mp in material_id:
    # start in the correct directory
    os.chdir(base_dir)

    # setup everything
    if not os.path.exists(os.path.join(calc_folder, mp)):
        # create directory
        os.makedirs(os.path.join(calc_folder, mp))

    if not os.path.exists(os.path.join(calc_folder, mp, "structure.pckl")):
        # download the structure with the Materials Project API
        # (this could change later on with we use different databases...)
        structure, name, metal_check = basic_utils.get_structure(api_key, mp)

        if metal_check:
            continue

        # save the structure to a file
        f = open(os.path.join(calc_folder, mp, "structure.pckl"), "wb")
        pickle.dump([structure, name], f)
        f.close()

    # create the job file and start the job
    filename = basic_utils.start_calc(
        base_dir,
        calc_setup,
        mp,
        script_name,
        file_name="htdf_job",
        ncores=ncores,
        memory=memory,
        calc_folder=calc_folder,
    )
