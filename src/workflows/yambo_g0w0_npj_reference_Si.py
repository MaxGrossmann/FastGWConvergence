"""
This workflows performs the reference calculation for the SOTA
algorithm. It assumes that the DFT calculation and yambo setup
was already done. The parameters here are adjusted to match to ones 
for Si from https://doi.org/10.1038/s41524-023-01027-2.
"""

# external imports
import sys

# local imports
import src.utils.yambo_runner as yambo_runner

# base directory
base_dir = str(sys.argv[1])

# number of cores
ncores = int(sys.argv[2])

# get the id from the args that are called with this script
id = str(sys.argv[3])

# convergence threshold (if conv_percent > 0 it is used instead of conv_thr...)
conv_thr = 0.010  # eV
conv_percent = 0  # % relative to the direct gap

# parameter step size
bnd_step = 200
cut_step = 4
edges = [200, 4, 800, 16]

# run the convergence
yambo_runner.yambo_run_gw_conv_npj_reference(
    ncores,
    edges=edges,
    bnd_step=bnd_step,
    cut_step=cut_step,
    conv_thr=conv_thr,
    conv_percent=conv_percent,
)
