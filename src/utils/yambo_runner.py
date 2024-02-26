"""
Functions that can start and restart yambo workflows.
Primarily used for the GW convergence calculations.
"""

# external imports
import os
import pickle
import shutil
import numpy as np

# local imports
from src.utils.calc_data_class import calc_data
import src.utils.qe_helper as qe_helper
import src.utils.qe_write as qe_write
import src.utils.qe_runner as qe_runner
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write
from src.utils.yambo_gw_conv_class import conv_data, conv_data_npj


def yambo_run_gw_conv_npj(
    calc_data_scf,
    n_bands,
    kppa,
    gw,
    base_dir,
    ncpu,
    grid_shift=0,
    edges=[200, 4, 800, 16],
    bnd_step=200,
    cut_step=4,
    conv_thr=0.01,
    conv_percent=1,
):
    """
    Starts a scf & nscf in qe, creates the a folder for a gw calculation and starts the npj/sota gw convergence algorithm.
    This function is useful to restart a convergence workflow if the number
    of bands needed to converge needs to be increased above the number used in the nscf step.
    INPUT:
        calc_data:      Class that contains the data for a qe calculation
        nbands:         Starting value for the maximum number of bands, this will be increased if too small
        kppa:           k-point density for the nscf calculation
        gw:             Convergence class used for restarting, else set to None
        base_dir:       Needed to find get the correct path to the pseudopotential files
        ncpu:           Number of cpu cores used for a mpi calls
        grid_shift:     How many times has the starting grid been shifted? (needed restarts)
        edges:          Edges of the grid used for the npj fit algorithm
        bnd_step:       Steps in the number of bands
        cut_step:       Steps in the cutoff
        conv_thr:       Convergence threshold for the direct gap
        conv_percent:   Convergence threshold in percent for the direct gap
    OUTPUT:
        conv_flag:      Flag, True if the convergence was successful
        gw:             Convergence class used for restarting, i.e. when the maximum number of bands needs to be increased
        filename_scf:   Filename of the scf calculation
        filename_nscf:  Filename of the nscf calculation
        pw_cutoff:      Cutoff used for the last nscf calculation
    """

    # adjust the number of bands to fit with the number of cores
    if n_bands < edges[2]:
        print(
            "Increasing n_bands to be compatible with the starting grid...", flush=True
        )
        n_bands = edges[2] + bnd_step
    n_bands = n_bands + n_bands % ncpu

    # unit cell volume to estimate the cutoff for the number of wanted bands
    vol = qe_helper.uc_vol_au(calc_data_scf.structure)

    # estimated the cutoff for the number of wanted bands
    # (the factor 1.5 is a lucky accident that seems to work somehow...???)
    pw_cutoff = np.max(
        [
            int(np.ceil(((1.5 * 8 * np.pi**2 * n_bands) / vol) ** (2 / 3))),
            calc_data_scf.pw_cutoff,
        ]
    )

    # run the scf with the given parameters
    calc_data_scf.pw_cutoff = pw_cutoff
    filename_scf = qe_runner.qe_pw_run(
        calc_data_scf, qe_write.write_scf, ncpu, kwargs={"symmorphic": True}
    )

    # create nscf calculation based on scf calculation
    calc_data_nscf = calc_data(
        calc_data_scf.structure,
        calc_data_scf.name,
        id=calc_data_scf.id,
        ibrav=calc_data_scf.ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=os.path.join(base_dir, "pseudo", "SG15"),
    )

    # try the nscf with a lot of bands, if it fails reduce the number of bands until it converges
    while True:
        filename_nscf = qe_runner.qe_pw_run(
            calc_data_nscf,
            qe_write.write_nscf_yambo,
            ncpu,
            kwargs={"n_bands": n_bands},
        )
        # check if the nscf crashed (happens when there are too many bands)
        f = open(filename_nscf + ".out", "r")
        nscf_out_str = f.read()
        f.close()
        if "JOB DONE." in nscf_out_str:
            print(f"NSCF converged with n_bands = {n_bands:d}!\n", flush=True)
            break
        else:
            print(f"\nNSCF not converged with n_bands = {n_bands:d} ...", flush=True)
            n_bands = n_bands - 2 * ncpu
            print(f"Trying again with n_bands = {n_bands:d} ...\n", flush=True)
            filename_nscf = qe_runner.qe_pw_run(
                calc_data_nscf,
                qe_write.write_nscf_yambo,
                ncpu,
                kwargs={"n_bands": n_bands},
            )

    # create a yambo subfolder
    if not os.path.exists(f"kppa{kppa:d}_npj"):
        os.mkdir(f"kppa{kppa:d}_npj")

    # p2y step with the output redirected to the yambo folder
    os.chdir(f"out/{calc_data_scf.id}.save")
    os.system(f"p2y -O ../../kppa{kppa:d}_npj/")

    # move to the subfolder for the yambo calculation
    os.chdir(f"../../kppa{kppa:d}_npj")
    print(f"Current directory: kppa{kppa:d}_npj", flush=True)

    # finish the setup
    os.system("yambo")
    path_to_rsetup = os.getcwd()

    # npj fit algorithm
    if not os.path.isdir("g0w0_npj"):
        os.mkdir("g0w0_npj")
    os.chdir("g0w0_npj")
    if gw is None:
        gw = conv_data_npj(
            ncpu,
            path_to_rsetup,
            conv_thr=conv_thr,
            edges=edges,
            bnd_step=bnd_step,
            cut_step=cut_step,
            cut_max=46,  # hard coded ... larger is computationally to expense
        )
    if gw is not None:
        gw.bnd_max = n_bands
    conv_flag = gw.run_convergence(conv_percent=conv_percent, grid_shift=grid_shift)

    # clean up the directory
    gw.convergence_cleanup()

    # plot the convergence path, clean up the class and save it
    if conv_flag:
        gw.plot_convergence()
        gw.delete_fit_arrays()
        with open("class_npj.pckl", "wb") as f:
            pickle.dump(gw, f)
    else:
        print("\nConvergence failed! Plot not created!\n")

    # back to the starting directory
    os.chdir("../../")

    # clear up the yambo directory
    if os.path.isfile(f"kppa{kppa:d}_npj/l_setup"):
        os.remove(f"kppa{kppa:d}_npj/l_setup")
        os.remove(f"kppa{kppa:d}_npj/r_setup")
    shutil.rmtree(f"kppa{kppa:d}_npj/SAVE")

    # if the convergence failed delete the scf file, nscf file and output folder from qe
    if not conv_flag:
        os.remove(filename_scf + ".in")
        os.remove(filename_scf + ".out")
        os.remove(filename_nscf + ".in")
        os.remove(filename_nscf + ".out")
        shutil.rmtree("out")

    return conv_flag, gw, filename_scf, filename_nscf, pw_cutoff


def yambo_run_gw_conv_npj_reference(
    ncpu,
    edges=[200, 4, 800, 16],
    bnd_step=200,
    cut_step=4,
    conv_thr=0.01,
    conv_percent=1,
    fftgvecs=None,
):
    """
    Simple function that starts a npj convergence calculation used for the reference calculation
    reproducing the results from Bonacci et al. (https://doi.org/10.1038/s41524-023-01027-2).
    INPUT:
        ncpu:           Number of cpu cores used for a mpi calls
        edges:          Edges of the grid used for the npj fit algorithm
        bnd_step:       Steps in the number of bands
        cut_step:       Steps in the cutoff
        conv_thr:       Convergence threshold for the direct gap
        conv_percent:   Convergence threshold in percent for the direct gap
        fftgvecs:       Cutoff for the FFT grid, only used for the C2 (diamond) reference calculation
    """

    # yambo folder
    if not os.path.isdir("g0w0_npj"):
        os.mkdir("g0w0_npj")
    os.chdir("g0w0_npj")

    # npj fit algorithm
    gw = conv_data_npj(
        ncpu,
        "../",
        conv_thr=conv_thr,
        edges=edges,
        bnd_step=bnd_step,
        cut_step=cut_step,
        cut_max=46,  # hard coded ... larger is computationally to expense
        fftgvecs=fftgvecs,
    )
    conv_flag = gw.run_convergence(conv_percent=conv_percent)

    # clean up the directory
    gw.convergence_cleanup()

    # plot the convergence path, clean up the class and save it
    if conv_flag:
        gw.plot_convergence()
        gw.delete_fit_arrays()
        with open("class_npj.pckl", "wb") as f:
            pickle.dump(gw, f)
    else:
        print("\nConvergence failed! Plot not created!\n")


def yambo_run_gw_conv_cs(
    calc_data_scf,
    n_bands,
    kppa,
    gw,
    base_dir,
    ncpu,
    bnd_start=200,
    bnd_step=100,
    cut_start=4,
    cut_step=4,
    conv_thr=0.01,
    ref_flag=False,
):
    """
    Starts a scf & nscf in qe, creates a folder for a gw calculation and starts the cs gw convergence algorithm.
    This function is useful to restart a convergence workflow if the number
    of bands needed to converge needs to be increased above the number used in the nscf step.
    INPUT:
        calc_data:      Class that contains the data for a qe calculation
        nbands:         Starting value for the maximum number of bands, this will be increased if too small
        kppa:           k-point density for the nscf calculation
        gw:             Convergence class used for restarting, else set to None
        base_dir:       Needed to find get the correct path to the pseudopotential files
        ncpu:           Number of cpu cores used for a mpi calls
        bnd_start:      Starting number of bands
        cut_start:      Starting cutoff
        bnd_step:       Steps in the number of bands
        cut_step:       Steps in the cutoff
        conv_thr:       Convergence threshold for the direct gap
        ref_flag:       Perform reference calculations with a very high number of bands and large cutoff
    OUTPUT:
        conv_flag:          Flag, True if the convergence was successful
        bnd_increase_flag:  Flag that indicates if the number of bands should be increased
        gw:                 Convergence class used for restarting, i.e. when the maximum number of bands needs to be increased
        filename_scf:       Filename of the scf calculation
        filename_nscf:      Filename of the nscf calculation
        pw_cutoff:          Cutoff used for the last nscf calculation
    """

    # adjust the number of bands to fit with the number of cores
    if n_bands < bnd_start:
        print(
            "Increasing n_bands to be compatible with the starting point...", flush=True
        )
        n_bands = bnd_start + 3 * bnd_step
    n_bands = n_bands + n_bands % ncpu

    # unit cell volume to estimate the cutoff for the number of wanted bands
    vol = qe_helper.uc_vol_au(calc_data_scf.structure)

    # estimated the cutoff for the number of wanted bands
    # (the factor 1.5 is a lucky accident that seems to work somehow...???)
    pw_cutoff = np.max(
        [
            int(np.ceil(((1.5 * 8 * np.pi**2 * n_bands) / vol) ** (2 / 3))),
            calc_data_scf.pw_cutoff,
        ]
    )

    # run the scf with the given parameters
    calc_data_scf.pw_cutoff = pw_cutoff
    filename_scf = qe_runner.qe_pw_run(
        calc_data_scf, qe_write.write_scf, ncpu, kwargs={"symmorphic": True}
    )

    # create nscf calculation based on scf calculation and run it
    calc_data_nscf = calc_data(
        calc_data_scf.structure,
        calc_data_scf.name,
        id=calc_data_scf.id,
        ibrav=calc_data_scf.ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=os.path.join(base_dir, "pseudo", "SG15"),
    )

    # try the nscf with a lot of bands, if it fails reduce the number of bands until it converges
    while True:
        filename_nscf = qe_runner.qe_pw_run(
            calc_data_nscf,
            qe_write.write_nscf_yambo,
            ncpu,
            kwargs={"n_bands": n_bands},
        )
        # check if the nscf crashed (happens when there are too many bands)
        f = open(filename_nscf + ".out", "r")
        nscf_out_str = f.read()
        f.close()
        if "JOB DONE." in nscf_out_str:
            print(f"NSCF converged with n_bands = {n_bands:d}!\n", flush=True)
            break
        else:
            print(f"\nNSCF not converged with n_bands = {n_bands:d} ...", flush=True)
            n_bands = n_bands - 4 * ncpu
            print(f"Trying again with n_bands = {n_bands:d} ...\n", flush=True)
            filename_nscf = qe_runner.qe_pw_run(
                calc_data_nscf,
                qe_write.write_nscf_yambo,
                ncpu,
                kwargs={"n_bands": n_bands},
            )

    # create a yambo subfolder
    if not os.path.exists(f"kppa{kppa:d}_cs"):
        os.mkdir(f"kppa{kppa:d}_cs")

    # p2y step with the output redirected to the yambo folder
    os.chdir(f"out/{calc_data_scf.id}.save")
    os.system(f"p2y -O ../../kppa{kppa:d}_cs/")

    # move to the subfolder for the yambo calculation
    os.chdir(f"../../kppa{kppa:d}_cs")
    print(f"Current directory: kppa{kppa:d}_cs", flush=True)

    # yambo setup
    os.system("yambo")
    path_to_rsetup = os.getcwd()

    # npj fit algorithm
    if not os.path.isdir("g0w0_cs"):
        os.mkdir("g0w0_cs")
    os.chdir("g0w0_cs")
    if gw is None:
        gw = conv_data(
            ncpu,
            path_to_rsetup,
            conv_thr=conv_thr,
            bnd_start=bnd_start,
            bnd_step=bnd_step,
            cut_start=cut_start,
            cut_step=cut_step,
            cut_max=46,  # hard coded ... larger is computationally to expense
            ref_flag=ref_flag,
        )
    if gw is not None:
        gw.bnd_max = n_bands
    conv_flag, bnd_increase_flag = gw.run_convergence()

    # clean up the directory
    gw.convergence_cleanup()

    # plot the convergence path and save the class
    if conv_flag:
        gw.plot_convergence()
        with open("class_cs.pckl", "wb") as f:
            pickle.dump(gw, f)
    else:
        print("\nConvergence failed! Plot not created!\n")

    # back to the starting directory
    os.chdir("../../")

    # clear up the yambo directory
    if os.path.isfile(f"kppa{kppa:d}_cs/l_setup"):
        os.remove(f"kppa{kppa:d}_cs/l_setup")
        os.remove(f"kppa{kppa:d}_cs/r_setup")
    shutil.rmtree(f"kppa{kppa:d}_cs/SAVE")

    # if the convergence failed delete the scf file, nscf file and output folder from qe
    if not conv_flag:
        os.remove(filename_scf + ".in")
        os.remove(filename_scf + ".out")
        os.remove(filename_nscf + ".in")
        os.remove(filename_nscf + ".out")
        shutil.rmtree("out")

    return conv_flag, bnd_increase_flag, gw, filename_scf, filename_nscf, pw_cutoff


def yambo_run_gw_conv_cs_reference(
    calc_data_scf,
    kppa,
    base_dir,
    ncpu,
    bnd_max=1200,
    cut_max=46,
    pw_cutoff=None,
):
    """
    Starts a scf & nscf in qe, creates the a folder for a gw calculation and runs it.
    The restart feature may be usable, but has not been tested here because it is just a copy of the
    other functions. For the reference calculation, the number of bands and the cutoff are set to very high values.
    For our paper, the number of bands is set to 1200 and the cutoff to 46 Ry. For the nscf calculation, the pw cutoff
    is increased until the nscf converges at 1200 bands. This is a little different from the other functions,
    where the number of bands is decreased until the nscf converges with the previously estimated cutoff.
    We know that this is not absolutely optimal, but it works.
    INPUT:
        calc_data:      Class that contains the data for a qe calculation
        kppa:           k-point density for the nscf calculation
        base_dir:       Needed to find get the correct path to the pseudopotential files
        ncpu:           Number of cpu cores used for a mpi calls
        bnd_max:        Large number of bands for the reference calculation
        cut_max:        Large cutoff for the reference calculation
        pw_cutoff:      If a good cutoff is known it can be given here
    OUTPUT:
        ref_gap:            Reference gap
        num_kpt:            Total number of k-points
        filename_scf:       Filename of the scf calculation
        filename_nscf:      Filename of the nscf calculation
        pw_cutoff:          Cutoff used for the last nscf calculation
    """

    # unit cell volume to estimate the cutoff for the number of wanted bands
    vol = qe_helper.uc_vol_au(calc_data_scf.structure)

    # estimated the cutoff for the number of wanted bands
    if pw_cutoff is None:
        pw_cutoff = np.max(
            [
                int(np.ceil(((1.5 * 8 * np.pi**2 * bnd_max) / vol) ** (2 / 3))),
                calc_data_scf.pw_cutoff,
            ]
        )

    # run the scf with the given parameters
    calc_data_scf.pw_cutoff = pw_cutoff
    filename_scf = qe_runner.qe_pw_run(
        calc_data_scf, qe_write.write_scf, ncpu, kwargs={"symmorphic": True}
    )

    # create nscf calculation based on scf calculation and run it
    calc_data_nscf = calc_data(
        calc_data_scf.structure,
        calc_data_scf.name,
        id=calc_data_scf.id,
        ibrav=calc_data_scf.ibrav,
        calc_type="nscf",
        pw_cutoff=pw_cutoff,
        kppa=kppa,
        pseudo=os.path.join(base_dir, "pseudo", "SG15"),
    )

    # try the nscf with alot of bands, if it fails reduce the number of bands until it converges
    while True:
        filename_nscf = qe_runner.qe_pw_run(
            calc_data_nscf,
            qe_write.write_nscf_yambo,
            ncpu,
            kwargs={"n_bands": bnd_max},
        )
        # check if the nscf crashed (happens when there are too many bands)
        f = open(filename_nscf + ".out", "r")
        nscf_out_str = f.read()
        f.close()
        if "JOB DONE." in nscf_out_str:
            print(
                f"NSCF converged with n_bands = {bnd_max:d}, "
                + f"cutoff = {calc_data_nscf.pw_cutoff:d}!\n",
                flush=True,
            )

            # we need to redo the scf and nscf with the same cutoff
            # otherwise the RIM breaks for unknown reasons
            if calc_data_nscf.pw_cutoff != calc_data_scf.pw_cutoff:
                os.remove(filename_scf + ".in")
                os.remove(filename_scf + ".out")
                os.remove(filename_nscf + ".in")
                os.remove(filename_nscf + ".out")
                shutil.rmtree("out")
                calc_data_scf.pw_cutoff = calc_data_nscf.pw_cutoff
                filename_scf = qe_runner.qe_pw_run(
                    calc_data_scf, qe_write.write_scf, ncpu, kwargs={"symmorphic": True}
                )
                filename_nscf = qe_runner.qe_pw_run(
                    calc_data_nscf,
                    qe_write.write_nscf_yambo,
                    ncpu,
                    kwargs={"n_bands": bnd_max},
                )

            # now go to the gw step
            pw_cutoff = calc_data_nscf.pw_cutoff
            break
        else:
            print(
                f"\nNSCF not converged with n_bands = {bnd_max:d} "
                + f"cutoff = {calc_data_nscf.pw_cutoff:d}!\n",
                flush=True,
            )
            os.remove(filename_nscf + ".in")
            os.remove(filename_nscf + ".out")
            calc_data_nscf.pw_cutoff += 5
            print(
                f"Trying again with a higher cutoff = {calc_data_nscf.pw_cutoff:d} ...\n",
                flush=True,
            )

    # create a yambo subfolder
    if not os.path.exists(f"kppa{kppa:d}_cs_ref"):
        os.mkdir(f"kppa{kppa:d}_cs_ref")

    # p2y step with the output redirected to the yambo folder
    os.chdir(f"out/{calc_data_scf.id}.save")
    os.system(f"p2y -O ../../kppa{kppa:d}_cs_ref/")

    # move to the subfolder for the yambo calculation
    os.chdir(f"../../kppa{kppa:d}_cs_ref")
    print(f"Current directory: kppa{kppa:d}_cs_ref", flush=True)

    # yambo setup
    os.system("yambo")

    # get the number of electrons in the unit cell
    num_elec = yambo_helper.get_num_electrons("r_setup")
    if num_elec & 0x1:
        raise Exception("Uneven number of electrons in the unit cell!")

    # get the total number of q-points
    num_kpt = yambo_helper.get_num_kpt("r_setup")

    # read the r_setup to find where the direct gap is situated
    kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters("r_setup")
    if kpt_bnd_idx[2] < num_elec / 2:
        print("Metallic states are present...", flush=True)
        kpt_bnd_idx[2] = int(num_elec / 2)
        kpt_bnd_idx[3] = int(num_elec / 2 + 1)

    # npj fit algorithm
    if not os.path.isdir("g0w0_cs_ref"):
        os.mkdir("g0w0_cs_ref")
    os.chdir("g0w0_cs_ref")
    # create the input file and start the calculation
    f_name = yambo_write.write_g0w0(
        bnd_max,
        cut_max,
        bnd_max,
        kpt_bnd_idx,
    )
    os.system(f"mpirun -np {ncpu} yambo -F {f_name}.in -J {f_name} -I ../")
    ref_gap = yambo_helper.get_minimal_gw_gap(f_name, kpt_bnd_idx)

    # clean up the yambo output
    files = os.listdir()
    for f in files:
        if os.path.isdir(f):
            shutil.rmtree(f)

    # back to the starting directory
    os.chdir("../../")

    # clear up the yambo directory
    if os.path.isfile(f"kppa{kppa:d}_cs_ref/l_setup"):
        os.remove(f"kppa{kppa:d}_cs_ref/l_setup")
        os.remove(f"kppa{kppa:d}_cs_ref/r_setup")
    shutil.rmtree(f"kppa{kppa:d}_cs_ref/SAVE")

    return ref_gap, num_kpt, filename_scf, filename_nscf, pw_cutoff
