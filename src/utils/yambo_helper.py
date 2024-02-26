"""
Here we store little functions that are 
a) Only necessary for yambo and 
b) Don't start calculations on their own
"""

# external imports
import re
import glob
import numpy as np


def get_gamma_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the k-point and band indices of the gap at the gamma point.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the bandstructure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the direct bandgap at the gamma point
    kpt_bnd_idx[2] = int(
        re.findall("\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the direct bandgap at the gamma point
    kpt_bnd_idx[0] = 1
    kpt_bnd_idx[1] = 1

    return kpt_bnd_idx


def get_direct_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the k-point and band indices of the direct gap and the gap value.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        direct_gap:         Value of the direct gap in eV
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the bandstructure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the direct bandgap
    kpt_bnd_idx[2] = int(
        re.findall("\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the direct bandgap
    kpt_bnd_idx[0] = int(
        re.findall(
            "\d+",
            re.findall(r"Direct Gap localized at k[ \t]+:[ \t]+\d+", setup_str)[0],
        )[0]
    )
    kpt_bnd_idx[1] = kpt_bnd_idx[0]

    # get the direct bandgap
    direct_gap = float(
        re.findall(
            "\d+.\d+", re.findall(r"Direct Gap[ \t]+:[ \t]+\d+.\d+", setup_str)[0]
        )[0]
    )

    return direct_gap, kpt_bnd_idx


def get_indirect_gap_parameters(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the k-point and band indices of the indirect gap and the gap value.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        indirect_gap:       Value of the indirect gap in eV
        kpt_bnd_idx:        kpt_idx1, kpt_idx2, bnd_idx1, bnd_idx2
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # array for gap location in the bandstructure
    kpt_bnd_idx = np.zeros(4, dtype=int)

    # get the vbm and cbm index of the indirect bandgap
    kpt_bnd_idx[2] = int(
        re.findall("\d+", re.findall(r"Filled Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )
    kpt_bnd_idx[3] = kpt_bnd_idx[2] + 1

    # get the k-point index of the indirect bandgap
    matches = re.findall(
        "\d+",
        re.findall(r"Indirect Gap between kpts[ \t]+:[ \t]+\d+[ \t]+\d+", setup_str)[0],
    )
    kpt_idx = [int(m) for m in matches]
    kpt_bnd_idx[0] = kpt_idx[0]
    kpt_bnd_idx[1] = kpt_idx[1]

    # get the indirect bandgap
    indirect_gap = float(
        re.findall(
            "\d+.\d+", re.findall(r"Indirect Gap[ \t]+:[ \t]+\d+.\d+", setup_str)[0]
        )[0]
    )

    return indirect_gap, kpt_bnd_idx


def get_num_electrons(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the number of electrons in the system.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        num_elec:           Number of electrons in the system
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get the number of electrons in the system
    num_elec = int(
        re.findall("\d+", re.findall(r"Electrons[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )

    return num_elec


def get_max_bands(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the maximum number of bands available.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        max_bands:          Maximum number of bands available
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get the maximum number of bands available
    max_bands = int(
        re.findall("\d+", re.findall(r"Bands[ \t]+:[ \t]+\d+", setup_str)[0])[0]
    )

    return max_bands


def get_num_kpt(path_to_rsetup):
    """
    Reads the r_setup file in the yambo folder from a given path path_to_rsetup.
    Returns the maximum number of bands available.
    INPUT:
        path_to_rsetup:     Path to the r_setup file
    OUTPUT:
        num_kpt:            Total number of k-points
    """

    # read the input file
    with open(path_to_rsetup, "r") as f:
        setup_str = f.read()

    # get number of k-points
    num_kpt = int(
        re.findall("\d+", re.findall(r"IBZ K-points :[ \t]+\d+", setup_str)[0])[0]
    )

    return num_kpt


def get_band_edges(f_name):
    """
    Reads the gw .qp output file from a gw convergence calculation at the direct gap.
    This functions always operates in the currect directory.
    Returns the energies of the band edges after the gw calculation.
    INPUT:
        f_name:         File name of the .qp output file
    OUTPUT:
        vbm:            Valence band maximum energy (eV)
        cbm:            Conduction band minimum energy (eV)
    """

    # parse the output file
    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    # calculate the band edge energies
    vbm = data[0, 2] + data[0, 3]
    cbm = data[1, 2] + data[1, 3]

    return vbm, cbm


def get_direct_gw_gap(f_name):
    """
    Reads the gw .qp output file from a gw convergence calculation at the direct gap.
    This functions always operates in the currect directory.
    Returns the new value of the direct gap after the gw calculation.
    INPUT:
        f_name:             File name of the .qp output file
    OUTPUT:
        direct_gap:         Direct gap energy (eV)
    """

    # parse the output file
    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    # calculate the direct gap
    direct_gap = (data[1, 2] + data[1, 3]) - (data[0, 2] + data[0, 3])

    return direct_gap


def get_minimal_gw_gap(f_name, kpt_bnd_idx):
    """
    Reads the gw .qp output file from a gw convergence calculation on the full q-grid.
    kpt_bnd_idx:    contains the k-point and band indicies where the minimal gap is located
    This functions always operates in the currect directory.
    Returns the new value of the minimal (indirect/direct) gap after the gw calculation.
    INPUT:
        f_name:             File name of the .qp output file
        kpt_bnd_idx:        Contains the k-point and band indicies where the minimal gap is located
    OUTPUT:
        min_gap:            Minimal band gap energy (eV)
    """

    # parse the output file
    data = np.loadtxt(f"o-{f_name}.qp", comments="#")

    # calculate the minimal gap
    k1 = data[data[:, 0] == kpt_bnd_idx[0]]
    vbm = np.sum(k1[k1[:, 1] == kpt_bnd_idx[2], :][0][2:4])
    k2 = data[data[:, 0] == kpt_bnd_idx[1]]
    cbm = np.sum(k2[k2[:, 1] == kpt_bnd_idx[3], :][0][2:4])
    min_gap = cbm - vbm

    return min_gap


def get_cut_from_report(f_name):
    """
    Reads the report output file from a gw convergence calculation.
    This functions always operates in the currect directory.
    Returns the actual cutoff energy used by yambo in mHa.
    INPUT:
        f_name:             File name of the .qp output file
    OUTPUT:
        cutsrc:             Actual cutoff energy in mHa
    """
    f_name = glob.glob(f"r-{f_name:s}*")[0]
    with open(f_name, "r") as f:
        fstr = f.read()
    res_block = re.findall(
        r"R(esponse block size in GW reduced to \d+[ \t]+RL \(\d+[ \t]+mHa\))", fstr
    )[0]
    cutsrc = int(re.findall(r"\d+", res_block)[1])  # in mHa

    return cutsrc
