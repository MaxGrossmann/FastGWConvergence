"""
Functions that write input files for YAMBO (but don't actually run them)
"""

# external imports
import os
import re
from textwrap import dedent


def write_g0w0(
    bnd_x,
    cutoff_screening,
    bnd_g,
    kpt_bnd_idx,
    ppa_energy=27.21138,
    f_name=None,
    fftgvecs=None,
    flag_2d=False,
):
    """
    Creates and adjusts the input file for a Yambo G0W0 calculation in the current directory.
    The function assumes that the save folder is one folder up.
    INPUT:
        bnd_x:              Number of bands included in the screening
        cutoff_screening:   Number G-vectors in the screening, i.e. the energy cutoff in Ry
        bnd_g:              Number of bands included in the greens function
        kpt_bnd_idx:        Range kpt1:kpt:2 & bnd1:bnd2 for which the qp energies are calculated (array of length 4)
        ppa_energy:         Plasmon pole imaginary energy in eV
        f_name:             Variable file name
        fftgvecs:           Currently only used for the C2 reference calculations
        flag_2d:            Flag for 2D materials (adjusts RIM)
    OUTPUT:
        f_name:             Name of the written input file
    """

    # input file name
    if f_name is None:
        f_name = f"g0w0_bndx_{bnd_x:d}_sc_{cutoff_screening:d}_bndg_{bnd_g:d}"

    # create the input file
    os.system(f"yambo -d -k hartree -g n -p p -r -F {f_name}.in -Q -I ../ -V qp")

    # read the input file
    with open(f"{f_name}.in", "r") as f:
        gw_str = f.read()

    # RIM 
    rim_str = dedent(
        r"""
        RandQpts=0 [ \t]+ \# \[RIM\] Number of random q-points in the BZ
        RandGvec= 1 [ \t]+ RL [ \t]+ \# \[RIM\] Coulomb interaction RS components
        CUTGeo= \"none\" [ \t]+ \# \[CUT\] Coulomb Cutoff geometry: box\/cylinder\/sphere\/ws\/slab X\/Y\/Z\/XY..
        """
    )
    if flag_2d:
        rim_rep_str = dedent(
            """
            RIM_W
            RandQpts= 5000024                # [RIM] Number of random q-points in the BZ
            RandGvec= 100 RL                 # [RIM] Coulomb interaction RS components
            CUTGeo= "slab Z"                 # [CUT] Coulomb Cutoff geometry: box/cylinder/sphere/ws/slab X/Y/Z/XY..
            RandGvecW= 15 RL
            """
        )
    else:
        rim_rep_str = dedent(
            """
            RandQpts= 5000024                # [RIM] Number of random q-points in the BZ
            RandGvec= 100 RL                 # [RIM] Coulomb interaction RS components
            CUTGeo= "none"                   # [CUT] Coulomb Cutoff geometry: box/cylinder/sphere/ws/slab X/Y/Z/XY..
            """
        )
    
    # adjust the input file
    gw_str = re.sub(rim_str, rim_rep_str, gw_str)
    gw_str = re.sub("#UseNLCC", "UseNLCC", gw_str)
    gw_str = re.sub(
        r"BndsRnXp\n[ \t]+[0-9]+[ \t]+\|[ \t]+[0-9]+[ \t]+\|",
        f"BndsRnXp\n   1 | {bnd_x:d} |",
        gw_str,
    )
    gw_str = re.sub(
        r"NGsBlkXp= 1[ \t]+RL", f"NGsBlkXp = {1000*cutoff_screening:d} mRy", gw_str
    )
    gw_str = re.sub(
        r"% LongDrXp\n[ \t]+[0-9]+.[0-9]+[ \t]+\|[ \t]+[0-9]+.[0-9]+[ \t]+\|[ \t]+[0-9]+.[0-9]+[ \t]+\|",
        f"% LongDrXp\n {1.0:.6f} | {1.0:.6f} | {1.0:.6f} | ",
        gw_str,
    )  # average screening along the [1 1 1]
    gw_str = re.sub(
        r"PPAPntXp= 27.21138         eV", f"PPAPntXp= {ppa_energy:.5f} eV", gw_str
    )
    gw_str = re.sub(
        r"GbndRnge\n[ \t]+[0-9]+[ \t]+\|[ \t]+[0-9]+[ \t]+\|",
        f"GbndRnge\n   1 | {bnd_g:d} |",
        gw_str,
    )
    gw_str = re.sub(r"GTermKind= \"none\"", r'GTermKind= "BG"', gw_str)
    gw_str = re.sub(
        r"\d+\|\d+\|\d+\|\d+\|",
        f"{kpt_bnd_idx[0]:d}|{kpt_bnd_idx[1]:d}|{kpt_bnd_idx[2]:d}|{kpt_bnd_idx[3]:d}|",
        gw_str,
    )
    gw_str = (
        gw_str.rsplit("\n", 4)[0] + "\n"
    )  # remove the last three lines as they are not needed
    gw_str += "NLogCPUs = 1\n"  # reduce the number of log-files

    # append the reduction of the fftgvecs if needed
    if fftgvecs is not None:
        gw_str += f"FFTGvecs = {int(fftgvecs):d} Ry"

    # write the adjusted input file
    with open(f"{f_name}.in", "w") as f:
        f.write(gw_str)

    return f_name


def write_g0w0_npj(
    bnd_x,
    cutoff_screening,
    bnd_g,
    kpt_bnd_idx,
    fftgvecs=None,
    flag_2d=False,
):
    """
    Creates and adjusts the input file for a Yambo G0W0 calculation in the current directory.
    (THIS INPUT FILE IS BASED ON https://www.nature.com/articles/s41524-023-01027-2)
    The function assumes that the save folder is one folder up.
    INPUT:
        bnd_x:              Number of bands included in the screening
        cutoff_screening:   Number G-vectors in the screening, i.e. the energy cutoff in Ry
        bnd_g:              Number of bands included in the greens function
        kpt_bnd_idx:        Range kpt1:kpt:2 & bnd1:bnd2 for which the qp energies are calculated (array of length 4)
        fftgvecs:           Currently only used for the C2 reference calculations
        flag_2d:            Flag for 2D materials (adjusts RIM)
    OUTPUT:
        f_name:             Name of the written input file
    """

    # input file name
    f_name = f"g0w0_bndx_{bnd_x:d}_sc_{cutoff_screening:d}_bndg_{bnd_g:d}"

    # print the file name
    print(f"\n{f_name:s}\n", flush=True)

    # create file string
    if flag_2d:
        gw_str = dedent(
            f"""\
        #
        # FGWC by MG2.  
        # YAMBO > 5.0 compatible
        # http://www.yambo-code.org
        #
        rim_cut
        dipoles
        gw0
        HF_and_locXC
        ppa
        RIM_W
        CUTGeo =   'slab Z'
        NLCC
        Chimod = 'hartree'
        % BndsRnXp
        1 | {bnd_x:d} |   
        %
        % GbndRnge
        1 | {bnd_g:d} |   
        %
        % LongDrXp
        1.0 | 1.0 | 1.0 |   
        %
        NGsBlkXp = {cutoff_screening:d} Ry
        % QPkrange
        {kpt_bnd_idx[0]:d} | {kpt_bnd_idx[1]:d} | {kpt_bnd_idx[2]:d} | {kpt_bnd_idx[3]:d} |   
        %
        RandGvec = 100 RL
        RandQpts = 5000024
        RandGvecW = 15 RL 
        DysSolver = 'n'
        GTermKind = 'BG'
        """
        )        
    else:
        gw_str = dedent(
            f"""\
        #
        # FGWC by MG2.  
        # YAMBO > 5.0 compatible
        # http://www.yambo-code.org
        #
        rim_cut
        dipoles
        gw0
        HF_and_locXC
        ppa
        NLCC
        Chimod = 'hartree'
        % BndsRnXp
        1 | {bnd_x:d} |   
        %
        % GbndRnge
        1 | {bnd_g:d} |   
        %
        % LongDrXp
        1.0 | 1.0 | 1.0 |   
        %
        NGsBlkXp = {cutoff_screening:d} Ry
        % QPkrange
        {kpt_bnd_idx[0]:d} | {kpt_bnd_idx[1]:d} | {kpt_bnd_idx[2]:d} | {kpt_bnd_idx[3]:d} |   
        %
        RandGvec = 100 RL
        RandQpts = 5000024 
        DysSolver = 'n'
        GTermKind = 'BG'
        """
        )

    # append the reduction of the fftgvecs if needed
    if fftgvecs is not None:
        gw_str += f"FFTGvecs = {int(fftgvecs):d} Ry"

    # write the adjusted input file
    with open(f"{f_name}.in", "w") as f:
        f.write(gw_str)

    return f_name
