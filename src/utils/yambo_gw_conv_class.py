"""
Here we store the classes that contain the two GW convergence algorithm (CS and NPJ/SOTA).    
"""

# external imports
import os
import scipy
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# local import
import src.utils.yambo_helper as yambo_helper
import src.utils.yambo_write as yambo_write

# setup for plots
plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rcParams.update({"mathtext.default": "regular"})
plt.rcParams["legend.markerscale"] = 0.5


class conv_data:
    """
    Class which contains data about the gw convergence in yambo
    ncores:             number of cores for each calculation
    path_to_rsetup:     relative path to the yambo setup file
    conv_thr:           convergence threshold for the direct gap
    bnd_start:          starting point for the number of bands
    cut_start:          starting point for the cutoff
    bnd_step:           steps in the number of bands
    cut_step:           steps in the cutoff
    cut_max:            maximum cutoff
    ref_flag:           whether the point with maximum convergence parameters should be calculated
    write_npj:          enable npj style input files
    flag_2d:            flag for 2D materials (adjusts RIM)
    """

    def __init__(
        self,
        ncores,
        path_to_rsetup,
        conv_thr=0.01,
        bnd_start=200,
        cut_start=6,
        bnd_step=50,
        cut_step=2,
        cut_max=46,
        ref_flag=False,
        write_npj=False,
        flag_2d=False,
    ):
        """
        Function that initializes all important parameters for the CS convergence algorithm.
        """
        # defaults
        self.ncores = ncores
        self.path_to_rsetup = path_to_rsetup
        self.bnd_start = bnd_start
        self.cut_start = cut_start
        self.conv_thr = conv_thr
        self.bnd_step = bnd_step
        self.cut_step = cut_step
        self.cut_max = cut_max
        self.ref_flag = ref_flag
        self.write_npj = write_npj
        self.flag_2d = flag_2d

        # complete the path to r_setup
        self.rsetup = os.path.join(self.path_to_rsetup, "r_setup")

        # get the number of electrons in the unit cell
        self.num_elec = yambo_helper.get_num_electrons(self.rsetup)
        if self.num_elec & 0x1:
            raise Exception("Uneven number of electrons in the unit cell!")

        # get the maximum number of bands
        self.bnd_max = yambo_helper.get_max_bands(self.rsetup)

        # get the total number of q-points
        self.num_kpt = yambo_helper.get_num_kpt(self.rsetup)

        # read the r_setup to find where the direct gap is situated
        self.kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters(self.rsetup)
        if self.kpt_bnd_idx[2] < self.num_elec / 2:
            print("Metallic states are present...", flush=True)
            self.kpt_bnd_idx[2] = int(self.num_elec / 2)
            self.kpt_bnd_idx[3] = int(self.num_elec / 2 + 1)

        # dummy for the input file names
        self.fn = []

        # initialize grid variable to keep the convergence progress for plotting
        self.grid = []

        # initialize to time the indiviual steps on the grid
        # (good for timing predictions for real high-throughput calculations)
        self.grid_time = []

    def run_convergence(self):
        """
        Main function that runs the CS convergence algorithm.
        All results are stored inside the class.
        """
        start_time = time.time()

        # reference calculation at the starting point (only done when starting fresh)
        if not self.grid:
            print("\nReference calculation at the starting point:", flush=True)
            if self.bnd_start > self.bnd_max:
                print("\nNot enough bands for the starting point...", flush=True)
                return False, True  # conv_flag, bnd_increase_flag
            if self.write_npj:
                f_name = yambo_write.write_g0w0_npj(
                    self.bnd_start, self.cut_start, self.bnd_start, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                )
            else:
                f_name = yambo_write.write_g0w0(
                    self.bnd_start, self.cut_start, self.bnd_start, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                )
            self.fn.append(f_name)
            step_time = time.time()
            os.system(
                f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
            )
            self.grid_time.append(time.time() - step_time)
            print(
                f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                flush=True,
            )
            self.gap_current = yambo_helper.get_direct_gw_gap(f_name)
            print(
                f"bnd_x = {self.bnd_start:d}, cutsrc = {self.cut_start:d}, bnd_g = {self.bnd_start:d}",
                flush=True,
            )
            print(f"gap_current = {self.gap_current:6f}", flush=True)
            self.grid.append([self.bnd_start, self.cut_start, self.gap_current])

            if self.ref_flag:
                # reference calculation with the maximum calculation parameters
                print(
                    "\nReference calculation at the maximum calculation parameters",
                    flush=True,
                )
                if self.write_npj:
                    f_name = yambo_write.write_g0w0_npj(
                        self.bnd_max, self.cut_max, self.bnd_max, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                else:
                    f_name = yambo_write.write_g0w0(
                        self.bnd_max, self.cut_max, self.bnd_max, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.ref_time = time.time() - step_time
                print(
                    f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                    flush=True,
                )
                self.ref_gap = yambo_helper.get_direct_gw_gap(f_name)
                print(
                    f"bnd_x = {self.bnd_max:d}, cutsrc = {self.cut_max:d}, bnd_g = {self.bnd_max:d}",
                    flush=True,
                )
                print(f"ref_gap = {self.ref_gap:6f}", flush=True)

        # handmade coordinate search type algorithm
        print("\nCoordinate search convergence:", flush=True)
        # parameters when doing the first start
        if len(self.grid_time) == 1:
            self.bnd = self.bnd_start
            self.cut = self.cut_start
            self.gap_diag = self.gap_current
            self.iter = 1
        diff_gap = 1  # some random value larger than the convergence threshold
        while True:
            if (self.bnd + self.bnd_step > self.bnd_max) and (
                self.cut + self.cut_step > self.cut_max
            ):
                print("\nMaximum calculation parameters were exceeded...", flush=True)
                return False, False  # conv_flag, bnd_increase_flag
            print(f"iter = {self.iter}", flush=True)
            while True:
                if self.bnd + self.bnd_step > self.bnd_max:
                    print("\nMaximum number of bands was exceeded...", flush=True)
                    return False, True  # conv_flag, bnd_increase_flag
                self.bnd += self.bnd_step
                if self.write_npj:
                    f_name = yambo_write.write_g0w0_npj(
                        self.bnd, self.cut, self.bnd, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                else:
                    f_name = yambo_write.write_g0w0(
                        self.bnd, self.cut, self.bnd, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.grid_time.append(time.time() - step_time)
                print(
                    f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                    flush=True,
                )
                new_gap = yambo_helper.get_direct_gw_gap(f_name)
                self.grid.append([self.bnd, self.cut, new_gap])
                delta_gap = np.abs(self.gap_current - new_gap)
                print(
                    f"bnd_x = {self.bnd:d}, cutsrc = {self.cut:d}, bnd_g = {self.bnd:d}",
                    flush=True,
                )
                print(f"gap_current = {self.gap_current:6f}", flush=True)
                print(f"new_gap     = {new_gap:6f}", flush=True)
                print(f"delta_gap   = {delta_gap:6f}", flush=True)
                self.gap_current = new_gap
                if delta_gap < self.conv_thr:
                    break
            while True:
                if self.cut + self.cut_step > self.cut_max:
                    print("\nMaximum cutoff was exceeded...", flush=True)
                    return False, False  # conv_flag, bnd_increase_flag
                self.cut += self.cut_step
                if self.write_npj:
                    f_name = yambo_write.write_g0w0_npj(
                        self.bnd, self.cut, self.bnd, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                else:
                    f_name = yambo_write.write_g0w0(
                        self.bnd, self.cut, self.bnd, self.kpt_bnd_idx, flag_2d=self.flag_2d,
                    )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.grid_time.append(time.time() - step_time)
                print(
                    f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                    flush=True,
                )
                new_gap = yambo_helper.get_direct_gw_gap(f_name)
                self.grid.append([self.bnd, self.cut, new_gap])
                delta_gap = np.abs(self.gap_current - new_gap)
                print(
                    f"bnd_x = {self.bnd:d}, cutsrc = {self.cut:d}, bnd_g = {self.bnd:d}",
                    flush=True,
                )
                print(f"gap_current = {self.gap_current:6f}", flush=True)
                print(f"new_gap     = {new_gap:6f}", flush=True)
                print(f"delta_gap   = {delta_gap:6f}", flush=True)
                self.gap_current = new_gap
                if delta_gap < self.conv_thr:
                    break
            print("\nDiagonal gap comparison:")
            print(f"gap_diag = {self.gap_diag:6f}", flush=True)
            print(f"gap_current = {self.gap_current:6f}", flush=True)
            diff_gap = np.abs(self.gap_diag - self.gap_current)
            print(f"diff_gap = {diff_gap:.6f}\n", flush=True)
            self.diag_diff = diff_gap
            if diff_gap <= self.conv_thr:
                break
            self.gap_diag = self.gap_current
            self.iter += 1

        # safety
        if self.bnd + self.bnd_step > self.bnd_max:
            print("\nReached maximum number of DFT bands!", flush=True)
        if self.cut + self.cut_step > self.cut_max:
            print("Reached maximum number of screening cutoff!", flush=True)

        # final report
        self.conv_thres = np.min([self.conv_thr, diff_gap])
        print(f"Reached convergence threshold: {self.conv_thres:.6f}", flush=True)
        print(f"Final gap = {self.gap_current:.6f}", flush=True)
        if self.ref_flag:
            print(f"Reference gap: {self.ref_gap:.6f}", flush=True)
            print(
                f"Difference to the reference gap: {np.abs(self.ref_gap-self.gap_current):.6f}"
            )
        self.final_point = np.array([self.bnd, self.cut, self.gap_current])
        print("\nFinal point:", flush=True)
        print(self.final_point, flush=True)

        # print the total number of gw calculation
        print(f"\nTotal number of GW calculation: {len(self.fn):d}")
        end_time = time.time()
        self.conv_time = end_time - start_time
        return True, False  # conv_flag, bnd_increase_flag

    def plot_convergence(self, figname="gw_conv_plt_cs.png"):
        """
        Function that plots the steps of the CS convergence algorithm.
        """
        print("\nPlotting the convergence steps...\n", flush=True)

        # turn the grid into a numpy array
        temp_grid = np.zeros([len(self.grid), 3])
        for i, g in enumerate(self.grid):
            temp_grid[i, :] = g
        self.grid = temp_grid

        # setup publication quality plot
        width = 4
        height = 4
        tick_size = int(width * 2.5)
        label_size = int(width * 3.0)
        plt.figure(figsize=(width, height), facecolor="w", dpi=300)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        ax = plt.gca()
        ax.set_xlabel(ax.get_xlabel(), size=label_size)
        ax.set_ylabel(ax.get_ylabel(), size=label_size)

        # scatter fit
        if self.ref_flag:
            rel_error = 100 * abs(self.grid[:, 2] - self.ref_gap) / self.ref_gap
        else:
            rel_error = 100 * abs(self.grid[:, 2] - self.grid[-1, 2]) / self.grid[-1, 2]
        fit = ax.scatter(
            self.grid[:, 0],
            self.grid[:, 1],
            75,
            c=rel_error,
            marker="o",
            zorder=10,
            cmap="viridis_r",
        )
        ax.plot(self.grid[:, 0], self.grid[:, 1], "r--", linewidth=2)

        # color bar
        cmap = plt.colorbar(fit, shrink=0.75, aspect=30, pad=0.05)
        cmap.set_label(
            r"$|\Delta E_{gap}^{\Gamma-\Gamma}|_\%$",
            rotation=90,
            fontsize=label_size,
        )
        cmap.ax.tick_params(labelsize=tick_size)

        # highlight the final point
        ax.scatter(
            self.grid[-1, 0],
            self.grid[-1, 1],
            100,
            c="r",
            marker="s",
            label=f"Converged = {np.round(self.grid[-1,2],6):.3f} eV",
            zorder=20,
        )

        if self.ref_flag:
            # highlight the final point
            ax.scatter(
                self.bnd_max,
                self.cut_max,
                100,
                c="k",
                marker="s",
                label=f"Reference = {np.round(self.ref_gap,6):.3f} eV",
                zorder=20,
            )

        # legend
        ax.legend(
            loc="upper left",
            fontsize=5,
        ).set_zorder(100)

        # acivate the grid
        ax.grid(color="k", linestyle="-", linewidth=0.75)

        # axis labels
        ax.set_xlabel(r"$N_b$")
        ax.set_ylabel(r"$G_{cut}$ (Ry)")

        # axis limits
        if self.ref_flag:
            ax.set_xlim([2, self.bnd_max + self.bnd_step])
        else:
            ax.set_xlim([2, np.max(self.grid[:, 0]) + self.bnd_step])

        ax.set_ylim([2, self.cut_max + 2])

        # ticks
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            which="both",
            width=1,
            length=4,
            direction="in",
        )
        ax.tick_params(
            labelbottom=True, labeltop=False, labelleft=True, labelright=False
        )

        # plot layout
        plt.tight_layout()

        # save the plot
        plt.savefig(figname, dpi=600)

    def convergence_cleanup(self):
        files = os.listdir()
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f)


# https://doi.org/10.1038/s41524-023-01027-2
class conv_data_npj:
    """
    Class which contains data about the gw convergence in yambo
    ncores:             number of cores for each calculation
    path_to_rsetup:     relative path to the yambo setup file
    conv_thr:           convergence threshold for the direct gap
    alpha:              positons the inbetween points on the grid
    edges:              grid edges: [bnd_1, cut_1, bnd_2, cut_2]
    bnd_step:           steps in the number of bands
    cut_step:           steps in the cutoff
    bnd_max:            maximum number of bands
    cut_max:            maximum cutoff
    conv_grad:          convergence threshold for the 1st derivatives
    conv_hessian:       convergence threshold for the coupling elements (mixed derivatives)
    max_grid_shift:     maximum number of allowed grid shifts
    write_npj:          enable npj style input files
    fftgvecs:           only used for C2 example
    flag_2d:            flag for 2D materials (adjusts RIM)
    """

    def __init__(
        self,
        ncores,
        path_to_rsetup,
        conv_thr=0.025,
        alpha=1 / 3,
        edges=[200, 4, 800, 16],
        bnd_step=200,
        cut_step=4,
        cut_max=46,
        conv_grad=5e-5,
        conv_hessian=1e-8,
        max_grid_shift=5,
        write_npj=False,
        fftgvecs=None,
        flag_2d=False,
    ):
        """
        Function that initializes all important parameters for the NPJ convergence algorithm.
        """
        # defaults
        self.ncores = ncores
        self.path_to_rsetup = path_to_rsetup
        self.conv_thr = conv_thr
        self.alpha = alpha
        self.edges = edges
        self.bnd_step = bnd_step
        self.cut_step = cut_step
        self.cut_max = cut_max
        self.conv_grad = conv_grad
        self.conv_hessian = conv_hessian
        self.max_grid_shift = max_grid_shift
        self.write_npj = write_npj
        self.fftgvecs = fftgvecs
        self.flag_2d = flag_2d
        
        # initializations
        self.gap = np.array(0)
        self.iter = []

        # complete the path to r_setup
        self.rsetup = os.path.join(self.path_to_rsetup, "r_setup")

        # get the number of electrons in the unit cell
        self.num_elec = yambo_helper.get_num_electrons(self.rsetup)
        if self.num_elec & 0x1:
            raise Exception("Uneven number of electrons in the unit cell!")

        # get the maximum number of bands
        self.bnd_max = yambo_helper.get_max_bands(self.rsetup)

        # get the total number of q-points
        self.num_kpt = yambo_helper.get_num_kpt(self.rsetup)

        # read the r_setup to find where the direct gap is situated
        self.kpt_bnd_idx = yambo_helper.get_gamma_gap_parameters(self.rsetup)
        if self.kpt_bnd_idx[2] < self.num_elec / 2:
            print("Metallic states are present...", flush=True)
            self.kpt_bnd_idx[2] = int(self.num_elec / 2)
            self.kpt_bnd_idx[3] = int(self.num_elec / 2 + 1)

        # dummy for the input file names
        self.fn = []

        # to time the indiviual steps on the grid
        # (good for timing predictions for real high-throughput calculations)
        self.grid_time = []

        # setup the array that contains the information for plotting
        self.grid_plot = np.array([])

    def create_grid(
        self,
        shift=[0, 0],
        old_grids=np.array([]),  # this is needed for the plot later on ...
    ):
        """
        This function creates the grid for the NPJ convergence algorithm.
        If a grid was computed previously, it should be provided through the
        old_grids variable, which will be updated with the new grid.
        This is needed for later plotting. The grid design is copied from:
        https://github.com/yambo-code/aiida-yambo.
        """
        # grid ranges
        b_min = self.edges[0] + shift[0] * self.bnd_step
        b_max = self.edges[2] + shift[0] * self.bnd_step
        g_min = self.edges[1] + shift[1] * self.cut_step
        g_max = self.edges[3] + shift[1] * self.cut_step

        # check if the maximum points are outside the maximum values
        if (b_max > self.bnd_max) or (g_max > self.cut_max):
            return False

        # create the four boundary point and two offset points
        p1 = [b_min, g_min]
        p2 = [b_max, g_min]
        p3 = [b_max, g_max]
        p4 = [b_min, g_max]
        p5 = [
            self.alpha * (b_max - b_min) + b_min,
            (1 - self.alpha) * (g_max - g_min) + g_min,
        ]
        p6 = [
            (1 - self.alpha) * (b_max - b_min) + b_min,
            self.alpha * (g_max - g_min) + g_min,
        ]

        # fix the position of the offset points to the grid
        space_b = np.arange(b_min, b_max + 1, self.bnd_step)
        space_g = np.arange(g_min, g_max + 1, self.cut_step)
        p5[0] = space_b[abs(space_b - p5[0]).argmin()]
        p5[1] = space_g[abs(space_g - p5[1]).argmin()]
        p6[0] = space_b[abs(space_b - p6[0]).argmin()]
        p6[1] = space_g[abs(space_g - p6[1]).argmin()]

        # keep track of the point numbers
        if not self.iter:
            iter_ref = 0
        else:
            iter_ref = self.iter[-1] + 1

        # create/update the array that contains the grid points with the gaps
        points = np.zeros([6, 3])  # [bnd_x, cutscr, gap]
        for i, p in enumerate([p1, p2, p3, p4, p5, p6]):
            points[i, :2] = p
            # skip points that where already calculated
            if old_grids.size > 0:
                temp_idx = (old_grids[:, 0] == p[0]) & (old_grids[:, 1] == p[1])
                if any(temp_idx):
                    points[i, 2] = old_grids[temp_idx][0][2]
                    print(
                        f"\nPoint p = ({p[0]:d}, {p[1]:d}) already calculated...\n"
                        + f"gap = {points[i,2]:.6f}",
                        flush=True,
                    )
                else:
                    if self.write_npj:
                        f_name = yambo_write.write_g0w0_npj(
                            p[0],
                            p[1],
                            p[0],
                            self.kpt_bnd_idx,
                            fftgvecs=self.fftgvecs,
                            flag_2d=self.flag_2d,
                        )
                    else:
                        f_name = yambo_write.write_g0w0(
                            p[0],
                            p[1],
                            p[0],
                            self.kpt_bnd_idx,
                            fftgvecs=self.fftgvecs,
                            flag_2d=self.flag_2d,
                        )
                    self.fn.append(f_name)
                    step_time = time.time()
                    os.system(
                        f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                    )
                    self.grid_time.append(time.time() - step_time)
                    print(
                        f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                        flush=True,
                    )
                    points[i, 2] = yambo_helper.get_direct_gw_gap(f_name)
                    print(f"gap = {points[i,2]:.6f}", flush=True)
                self.iter.append(iter_ref)
            else:
                if self.write_npj:
                    f_name = yambo_write.write_g0w0_npj(
                        p[0],
                        p[1],
                        p[0],
                        self.kpt_bnd_idx,
                        fftgvecs=self.fftgvecs,
                        flag_2d=self.flag_2d,
                    )
                else:
                    f_name = yambo_write.write_g0w0(
                        p[0],
                        p[1],
                        p[0],
                        self.kpt_bnd_idx,
                        fftgvecs=self.fftgvecs,
                        flag_2d=self.flag_2d,
                    )
                self.fn.append(f_name)
                step_time = time.time()
                os.system(
                    f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../"
                )
                self.grid_time.append(time.time() - step_time)
                print(
                    f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
                    flush=True,
                )
                points[i, 2] = yambo_helper.get_direct_gw_gap(f_name)
                print(f"gap = {points[i,2]:.6f}", flush=True)
                self.iter.append(iter_ref)

        # append the old grid for plotting the convergence path
        if old_grids.size > 0:
            old_grids_len = len(old_grids)
            points_plot = np.zeros([len(old_grids) + 6, 3])  # [bnd_x, cutscr, gap]
            points_plot[:old_grids_len, :] = old_grids
            for i in range(6):
                points_plot[old_grids_len + i, :] = points[i, :]
        else:
            points_plot = points
        self.grid_plot = points_plot  # for plotting later on we keep all the grids

        # just use the new grid every time the grid is shifted
        # (this algorithm just throws away the data of the old grid...)
        self.grid = points
        return True

    def append_point(self, point):
        """
        Function that performs a new GW calculation at a given point
        and appends the result to the grid.
        """
        # convert to integers so the file name is printed correctly
        point = np.array([int(p) for p in point])

        # create a new grid with one more point and add the old one
        new_grid = np.zeros([self.grid.shape[0] + 1, self.grid.shape[1]])
        for i in range(self.grid.shape[0]):
            new_grid[i, :] = self.grid[i, :]
        new_grid[-1, :2] = point
        new_grid_plot = np.zeros([self.grid_plot.shape[0] + 1, self.grid_plot.shape[1]])
        for i in range(self.grid_plot.shape[0]):
            new_grid_plot[i, :] = self.grid_plot[i, :]
        new_grid_plot[-1, :2] = point

        # calculate the new point and update the grid
        if self.write_npj:
            f_name = yambo_write.write_g0w0_npj(
                point[0],
                point[1],
                point[0],
                self.kpt_bnd_idx,
                fftgvecs=self.fftgvecs,
                flag_2d=self.flag_2d,
            )
        else:
            f_name = yambo_write.write_g0w0(
                point[0],
                point[1],
                point[0],
                self.kpt_bnd_idx,
                fftgvecs=self.fftgvecs,
                flag_2d=self.flag_2d,
            )
        self.fn.append(f_name)
        step_time = time.time()
        os.system(f"mpirun -np {self.ncores} yambo -F {f_name}.in -J {f_name} -I ../")
        self.grid_time.append(time.time() - step_time)
        print(
            f"Actual cutoff: {2*yambo_helper.get_cut_from_report(f_name):d} mRy\n",
            flush=True,
        )
        new_gap = yambo_helper.get_direct_gw_gap(f_name)
        new_grid[-1, 2] = new_gap
        new_grid_plot[-1, 2] = new_gap
        self.grid = new_grid
        self.grid_plot = new_grid_plot
        print(f"Calculated gap at the new point: {self.grid[-1, 2]:.6f}\n", flush=True)
        self.iter.append(self.iter[-1] + 1)

    def fit2d(self, alpha=1, beta=1):
        """
        Function for the 2D fit of the convergence surface.
        Practically copied from the aiida yambo repository:
        https://github.com/yambo-code/aiida-yambo.
        """
        # fit functions for the convergence surface
        f = lambda x, a, b, c, d: (a / (x[0] ** alpha) + b) * (c / (x[1] ** beta) + d)

        # wrong definitions of the fit function derivatives from the aiida yambo repository
        """
        fx = lambda x, a, c, d: (
            (-alpha * a / (x[0] ** (alpha + 1))) * (c / (x[1]) + d)
        )
        fy = lambda x, a, b, c: (
            (a / (x[0]) + b) * (-beta * c / (x[1] ** (beta + 1)))
        )
        """

        # corrected definitions of the fit function derivatives
        fx = lambda x, a, c, d: (
            (-alpha * a / (x[0] ** (alpha + 1))) * (c / (x[1] ** beta) + d)
        )
        fy = lambda x, a, b, c: (
            (a / (x[0] ** alpha) + b) * (-beta * c / (x[1] ** (beta + 1)))
        )

        # off-diagonal elements of the hessian matrix of the fit function
        fxy = lambda x, a, c: (
            (-alpha * a / (x[0] ** (alpha + 1))) * (-beta * c / (x[1] ** (beta + 1)))
        )

        # fit data
        xdata, ydata = (
            np.array((self.grid[:, 0], self.grid[:, 1])),
            self.grid[:, 2],
        )

        # fit the convergence surface
        popt, _ = scipy.optimize.curve_fit(
            f,
            xdata=xdata,
            ydata=ydata,
            sigma=1 / (xdata[0] * xdata[1]),
            bounds=(
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf],
            ),
        )

        # convergence critierum
        mae_int = np.average(
            (
                abs(
                    f(
                        xdata,
                        popt[0],
                        popt[1],
                        popt[2],
                        popt[3],
                    )
                    - ydata
                )
            ),
            weights=xdata[0] * xdata[1],
        )
        self.mae_fit = mae_int

        # get the extrapolated gap value
        self.extra = popt[1] * popt[3]

        # new grid to extrapolate the function far from the starting grid
        self.x_fit = np.arange(min(xdata[0]), max(xdata[0]) * 10, self.bnd_step)
        self.y_fit = np.arange(min(xdata[1]), max(xdata[1]) * 10, self.cut_step)

        # obtain the derivatives
        self.zx_fit = fx(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[2], popt[3])
        self.zy_fit = fy(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[1], popt[2])
        self.zxy_fit = fxy(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[2])

        # get the fit function values on the new grid
        self.x_fit, self.y_fit = np.meshgrid(self.x_fit, self.y_fit)
        self.z_fit = f(
            np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[1], popt[2], popt[3]
        )

        # check where the convergence conditions for the derivatives are meet
        self.condition_conv_calc = np.where(
            (np.abs(self.zx_fit) < self.conv_grad)
            & (np.abs(self.zy_fit) < self.conv_grad)
            & (np.abs(self.zxy_fit) < self.conv_hessian)
        )

        # save this fit to the old fit variable for later
        self.old_x_fit = self.x_fit
        self.old_y_fit = self.y_fit
        self.old_z_fit = self.z_fit

        # if no points match the convergence critieria return
        print(
            "Number of points that fullfil the derivative criteria: "
            + f"{len(self.x_fit[self.condition_conv_calc]):d}",
            flush=True,
        )

        if len(self.x_fit[self.condition_conv_calc]) == 0:
            return False

        # obtain a new grid where the estimated converged direct gap is situated
        b = max(max(xdata[0]), self.x_fit[self.condition_conv_calc][0] * 1.25)
        g = max(max(xdata[1]), self.y_fit[self.condition_conv_calc][0] * 1.25)

        # fit on the new grid with derivatives
        self.x_fit = np.arange(min(xdata[0]), b + 1, self.bnd_step)
        self.y_fit = np.arange(min(xdata[1]), g + 1, self.cut_step)
        self.z_fit = f(
            np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[1], popt[2], popt[3]
        )
        self.zx_fit = fx(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[2], popt[3])
        self.zy_fit = fy(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[1], popt[2])
        self.zxy_fit = fxy(np.meshgrid(self.x_fit, self.y_fit), popt[0], popt[2])
        self.x_fit, self.y_fit = np.meshgrid(self.x_fit, self.y_fit)

        return True

    def get_best_fit(
        self,
        power_laws=[1, 2],
    ):
        """
        Function that finds the best fit for the convergence surface trying
        different power laws for the exponents. This function is practically copied from:
        https://github.com/yambo-code/aiida-yambo.
        """
        # find the best initial fit parameters for the exponents
        error = 10  # random initial value
        for i in power_laws:
            for j in power_laws:
                print(f"\nTrying: alpha = {i:.2f}, beta = {j:.2f}", flush=True)
                self.fit2d(
                    alpha=i,
                    beta=j,
                )
                print(f"MAE = {self.mae_fit:.6f}", flush=True)
                if self.mae_fit < error:
                    ii, jj = i, j
                    error = self.mae_fit
        print(f"\nBest power law: alpha = {ii:.2f}, beta = {jj:.2f}\n", flush=True)

        # get the best initial fit
        self.check_passed = self.fit2d(
            alpha=ii,
            beta=jj,
        )

    def suggest_next_point(self):
        """
        Function that suggests the next point which should be calculated,
        based on the fit of the convergence surface.
        """
        # choose a reference
        if self.ref == "extra":
            reference = self.extra
        else:
            reference = self.z_fit[-1, -1]
        print(
            "Extrapolated gap value from fit parameters: " + f"{self.extra:.6f} eV",
            flush=True,
        )
        print(
            "Extrapolated gap value at points that fulfill the derivative criteria: "
            + f"{reference:.6f} eV",
            flush=True,
        )

        # converge with percentages?
        if self.conv_percent > 0:
            self.conv_thr = self.conv_percent / 100 * reference

        # find a point that satisfies the convergence condition
        self.discrepancy = np.round(
            abs(reference - self.z_fit),
            abs(int(np.round(np.log10(self.conv_thr), 0))),
        )
        self.condition = np.where((self.discrepancy <= self.conv_thr))
        self.next_bnd_x, self.next_cutsrc = (
            self.x_fit[self.condition][0],
            self.y_fit[self.condition][0],
        )
        self.ref_gap = self.z_fit[self.condition][0]

        # dictionary that contains the relevant information for the next step
        self.next_step = {
            "Nb": self.next_bnd_x,
            "Gc": self.next_cutsrc,
            "gap": self.ref_gap,
            "ref_gap": reference,
            "new_grid": False,
            "already_computed": False,
            "conv_thr": self.conv_thr,
            "conv_percent": self.conv_percent,
        }

        # is the suggested point outside the grid?
        if self.next_step["Nb"] > self.bnd_max or self.next_step["Gc"] > self.cut_max:
            self.next_step["new_grid"] = True

        # was the suggested point already computed?
        if (self.next_step["Nb"] in self.grid_plot[:, 0]) and (
            self.next_step["Gc"]
            in self.grid_plot[np.where(self.grid_plot[:, 0] == self.next_step["Nb"]), 1]
        ):
            self.next_step["already_computed"] = True
            temp_idx = (self.grid_plot[:, 0] == self.next_step["Nb"]) & (
                self.grid_plot[:, 1] == self.next_step["Gc"]
            )
            self.next_step["already_computed_gap"] = self.grid_plot[temp_idx][0][2]

        # messages
        print("\nNext step data:", flush=True)
        print(self.next_step, flush=True)

        return self.next_step

    def convergence_step(self, control):
        """
        This function performs one loop of the NPJ convergence algorithm.
        If it fails anywhere during the process, it returns the control dictionary.
        """
        # if we just want to fit again skip this
        # (previously calculated point is within the tolerance of the suggested fit point)
        if not control["fit_flag"]:
            control["grid_flag"] = self.create_grid(
                shift=[control["shift_val"], control["shift_val"]],
                old_grids=self.grid_plot,
            )

        # breaks if a point on the grid exceeds the maximum parameter values
        if not control["grid_flag"]:
            return control

        # find the best initial fit parameters for the exponents
        self.get_best_fit()

        # check if the fit is not reliable and shift the grid
        if self.mae_fit > 5 * self.conv_thr:
            print("Fit is not reliable. Shifting the grid...", flush=True)
            control["shift_val"] = control["shift_val"] + 1
            control["fit_flag"] = False
            return control

        # if no point fullfils the derivative conditions break and shift the grid
        if not self.check_passed:
            print(
                "No point fullfils the derivative condtions. Shifting the grid...",
                flush=True,
            )
            control["shift_val"] = control["shift_val"] + 1
            control["fit_flag"] = False
            return control

        # get the next point
        self.suggest_next_point()

        # check if the point exceeds the maximum parameter values
        if self.next_step["new_grid"]:
            print(
                "Suggest point is outside the maximum parameter values. Shifting the grid...",
                flush=True,
            )
            control["shift_val"] = control["shift_val"] + 1
            control["fit_flag"] = False
            return control

        # check if the point was already computed
        if self.next_step["already_computed"]:
            print(
                "\nSuggested point was already computed:",
                flush=True,
            )
            # check if the fit value and the calculate gap are within the tolerance
            if (
                np.abs(self.next_step["already_computed_gap"] - self.next_step["gap"])
                < self.conv_thr
            ):
                print(
                    "\nFit and already computed value are within the tolerance!",
                    flush=True,
                )
                self.final_point = np.array(
                    [
                        self.next_step["Nb"],
                        self.next_step["Gc"],
                        self.grid[-1, 2],
                    ]
                )
                control["conv_flag"] = True
                return control
            # else shift the grid because the fit is bad
            else:
                print(
                    "Fit and already computed value are are not within the tolerance\n"
                    + "Shifting the grid...",
                    flush=True,
                )
                control["shift_val"] = control["shift_val"] + 1
                control["fit_flag"] = False
                return control

        # calculate the suggested point
        self.append_point([self.next_step["Nb"], self.next_step["Gc"]])

        # check if the gap at the point is close enough to the fit
        if np.abs(self.grid[-1, 2] - self.next_step["gap"]) <= self.conv_thr:
            print(
                "New point is within the tolerance of the reference gap:\n"
                + "Fitting again with the additional point...",
                flush=True,
            )
            control["fit_flag"] = True
            return control
        else:
            print(
                "New point didnt meet the convergence criteria: Shifting the grid...",
                flush=True,
            )
            control["shift_val"] = control["shift_val"] + 1
            control["fit_flag"] = False
            return control

    def run_convergence(
        self,
        grid_shift=0,
        conv_percent=0,
        gap_ref=None,
    ):
        """
        Function that controls the NPJ convergence algorithm.
        The "convergence_step" function is called until convergence is reached
        or the algorithm fails when the maximum number of grid shifts or the
        parameter bounds are reached. The convergence algorithm is controlled
        by the control dictionary. Each time the "convergence_step" function
        is called, the control dictionary is updated, returned and processed by this function.
        """
        # initialize
        start_time = time.time()
        self.ref = gap_ref  # "extra" -> extrapolated gap is the reference
        self.conv_percent = conv_percent  # only used if conv_percent > 0
        self.shift_flag = (
            False  # set to True if the maximum number of grid shifts is exceeded
        )
        self.shift_val = (
            grid_shift  # useful for restarts after the number of bands was increased
        )

        # dictionary for the control
        control = {
            "shift_val": self.shift_val,
            "fit_flag": False,
            "grid_flag": True,
            "conv_flag": False,
        }

        while True:
            # print control dictionary
            print(f"\n{control}\n", flush=True)

            # perform one run through the flowchart
            control = self.convergence_step(control)

            # check if a point on the grid exceeds the maximum parameter values
            if not control["grid_flag"]:
                print(
                    "Grid limit is reached! Try increasing the bnd_max...",
                    flush=True,
                )
                self.shift_val = control["shift_val"]
                self.final_point = np.array([0, 0, 0])
                return False

            # check if the maximum number of grid shifts is reached
            if control["shift_val"] > self.max_grid_shift:
                print(
                    "Maximum number of grid shifts exceeded! Stopping the convergence...",
                    flush=True,
                )
                self.shift_flag = True
                self.shift_val = control["shift_val"]
                self.final_point = np.array([0, 0, 0])
                return False

            # check if convergence is achieved
            if control["conv_flag"]:
                print("\nConvergence achived!", flush=True)
                break

        # get the total time needed to achieve convergence
        end_time = time.time()
        self.conv_time = end_time - start_time

        # print the total number of gw calculation
        print(f"\nTotal number of GW calculation: {len(self.fn):d}")
        return True

    def plot_convergence(self, figname="gw_conv_plt_npj_algo.png"):
        """
        Function that plots the steps of the NPJ convergence algorithm
        in the style of Bonacci et al. (https://doi.org/10.1038/s41524-023-01027-2).
        """
        print("\nPlotting the convergence steps...\n", flush=True)

        # setup publication quality plot
        width = 4
        height = 4
        tick_size = int(width * 2.5)
        label_size = int(width * 3.0)
        plt.figure(figsize=(width, height), facecolor="w", dpi=300)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        ax = plt.gca()
        ax.set_xlabel(ax.get_xlabel(), size=label_size)
        ax.set_ylabel(ax.get_ylabel(), size=label_size)

        # scatter fit
        idx_plt = np.where(
            (self.x_fit >= self.edges[0] + self.bnd_step)
            & (self.y_fit >= self.edges[1] + self.cut_step)
        )
        rel_error = 100 * abs(self.z_fit - self.extra) / self.extra
        fit = ax.scatter(
            self.x_fit[idx_plt],
            self.y_fit[idx_plt],
            75,
            c=rel_error[idx_plt],
            marker="o",
            zorder=10,
            cmap="viridis_r",
        )

        # color bar
        cmap = plt.colorbar(fit, shrink=0.75, aspect=30, pad=0.05)
        cmap.set_label(
            r"$|\Delta E_{gap}^{\Gamma-\Gamma}|_\%$",
            rotation=90,
            fontsize=label_size,
        )
        cmap.ax.tick_params(labelsize=tick_size)

        # plot all simulation points
        ax.scatter(
            self.grid_plot[:, 0],
            self.grid_plot[:, 1],
            90,
            c="k",
            marker="s",
            label="Simulations",
            zorder=15,
        )

        # useful to identify grids and suggested points
        iterations = np.array(self.iter)
        num_iter = len(np.unique(iterations))

        # show the grids
        grid_c = [
            "tab:gray",
            "tab:purple",
            "tab:olive",
            "tab:pink",
            "tab:brown",
            "tab:cyan",
        ]
        grid_i = 1
        c = 1
        for i in range(num_iter):
            if np.sum(iterations == i) == 6:
                temp_grid = self.grid_plot[iterations == i, :2]
                start_point = np.array(
                    [np.min(temp_grid[:, 0]), np.min(temp_grid[:, 1])]
                )
                end_point = np.array([np.max(temp_grid[:, 0]), np.max(temp_grid[:, 1])])
                wh = end_point - start_point
                ax.add_patch(
                    Rectangle(
                        start_point,
                        wh[0],
                        wh[1],
                        alpha=1 / 3,
                        facecolor=grid_c[grid_i - 1],
                        edgecolor=None,
                        label=f"Grid {grid_i:d}",
                    )
                )
                grid_i += 1

            elif np.sum(iterations == i) == 1:
                ax.scatter(
                    self.grid_plot[iterations == i, 0],
                    self.grid_plot[iterations == i, 1],
                    100,
                    marker="s",
                    label=f"Iteration {c:d}",
                    zorder=20,
                )
                c += 1

        # highlight the final point in red
        ax.scatter(
            self.final_point[0],
            self.final_point[1],
            45,
            c="r",
            marker="s",
            label=f"Converged = {np.round(self.final_point[2],3):.3f} eV",
            zorder=20,
        )

        # legend
        ax.legend(
            loc="upper left",
            fontsize=5,
        ).set_zorder(100)

        # acivate the grid
        ax.grid(linewidth=0.75, zorder=50)

        # axis labels
        ax.set_xlabel(r"$N_b$")
        ax.set_ylabel(r"$G_{cut}$ (Ry)")

        # axis limits
        ax.set_xlim(
            [
                self.edges[0] - self.bnd_step,
                np.max(self.grid_plot[:, 0]) + self.bnd_step,
            ]
        )
        ax.set_ylim([2, self.cut_max + 2])

        # ticks
        ax.tick_params(
            bottom=True,
            top=True,
            left=True,
            right=True,
            which="both",
            width=1,
            length=4,
            direction="in",
        )
        ax.tick_params(
            labelbottom=True, labeltop=False, labelleft=True, labelright=False
        )

        # adjust the tick density at the x-axis
        ax.set_xticks(ax.get_xticks()[::2])

        # plot layout
        plt.tight_layout()

        # save the plot
        plt.savefig(figname, dpi=600)

        # clear the axis and figure
        plt.cla()
        plt.clf()

    def convergence_cleanup(self):
        """
        Function that cleans up the directory after the convergence algorithm.
        All directories are removed (QE database and YAMBO database).
        """
        files = os.listdir()
        for f in files:
            if os.path.isdir(f):
                shutil.rmtree(f)

    def delete_fit_arrays(self):
        """
        Function that deletes the fit arrays to reduce the
        file size of the pickled class.
        """
        self.old_x_fit = 0
        self.old_y_fit = 0
        self.old_z_fit = 0
        self.x_fit = 0
        self.y_fit = 0
        self.z_fit = 0
        self.zx_fit = 0
        self.zy_fit = 0
        self.zxy_fit = 0
        self.discrepancy = 0
        self.condition = 0
