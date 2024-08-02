# Fast GW Convergence (FGWC)

The code and data in this repository correspond to the research results published in the paper "[A robust, simple, and efficient convergence workflow for GW calculations](https://doi.org/10.1038/s41524-024-01311-9)". When using this repository, please cite the paper as: Großmann, M., Grunert, M. & Runge, E. A robust, simple, and efficient convergence workflow for GW calculations. npj Comput Mater 10, 135 (2024). https://doi.org/10.1038/s41524-024-01311-9

This code base was created to converge G0W0 calculations in a robust, simple and efficient way. 
These workflows are designed to analyze the convergence behavior. 
If you just want to converge your material system, use the *yambo_g0w0_conv* workflow after the *qe_convergence_SG15* workflow. 
You may need to customize the *start_calc* function in *src/utitls/basic_utils* for your supercomputing environment or set *calc_setup = "local"* in *main.py* to perform single calculations on the current interactive node.

**REQUIREMENTS**
We give the exact version we used because we found that newer versions of certain packages break the code. Please use them to make sure everything works as intended.

- Python 3.10.13 (with the following packages installed into the base enviroment)
    - ase 3.22.1
    - mp-api 0.41.1
    - numpy 1.26.4
    - scipy 1.12.0
    - pymatgen 2024.3.1
    - ipython 8.20.0
    - matplotlib 3.8.3
- Quantum ESPRESSO 7.1 ("/bin" directory loaded into the path, compiled for MPI)
- YAMBO 5.1 ("/bin" directory loaded into the path, compiled for MPI)

**SETUP/USAGE**

Create a file named *api_key.py* in the main directory that contains the following variable: api_key = *YOUR_API_KEY*, where *YOUR_API_KEY* can be found at https://next-gen.materialsproject.org/api.\
Run the *main.py* script in the main directory with the desired workflows and settings.
Possible workflows can be found in */src/workflows*.

**REFERENCE CALCULATION**

To make sure that our implementation of the algorithm from https://doi.org/10.1038/s41524-023-01027-2 by Bonacci et al. [1] works, we performed reference calculations, which can be found in the *ref* folder.
We tested diamond (mp-66), silicon (mp-149) and hBN (mp-984) using the newer 1.2 and older 1.0 versions of the SG15 pseudopotentials. We used the Quantum ESPRESSO input files provided by Bonacci et al. (https://doi.org/10.24435/materialscloud:6w-qh) to ensure that the G0W0 calculations start from the same starting point. The calculations for ZnO and rutile-TiO2 were too expensive for our local cluster to complete in a reasonable time due to the high-density k-grids used in these examples. For this reason, we have only tested the fit using the quasiparticle energies from the output files available at https://doi.org/10.24435/materialscloud:6w-qh. These can be found in the *ref* directory. The fits and results for ZnO and rutile-TiO2 can be found in the notebooks in the *ref* directory. Additional information on the reference calculation can be found in the Supplemental Information of the paper.

[1] Bonacci, M., Qiao, J., Spallanzani, N. et al. Towards high-throughput many-body perturbation theory: efficient algorithms and automated workflows. npj Comput Mater 9, 74 (2023). https://doi.org/10.1038/s41524-023-01027-2

**NAMING CHANGES**

The state-of-the-art (SOTA) algorithm mentioned in the paper is labeled "npj" instead of "sota" in the code for historical reasons.

**CLASS DATA AND PLOTS IN THE */calc/* DIRECTORY**

For each convergence calculation for each k-grid of every material a *class_cs.pckl*/*class_npj.pckl* file was created. 
These files contain all information about the convergence calculations and are based on the classes defined in */src/utils/yambo_gw_conv_class.py*.
Additonally a figure (*gw_conv_plt_cs.png*/*gw_conv_plt_npj.png*) is created, showing the convergence path. The colorbar in the CS plots shows
the relative difference between the direct band gap at every point (N<sub>b</sub>, G<sub>cut</sub>) and the final convergence point. For the NPJ (SOTA) algorithm
the colorbar shows the relative difference between the direct band gap at every point (N<sub>b</sub>, G<sub>cut</sub>) and the gap extrapolated fromt the final fit parameters. In both cases these quantities are abbreviated with |&Delta;E<sup>&Gamma;-&Gamma;</sup><sub>gap</sub>|<sub>%</sub>.
(This also applies to the reference calculation describe above.)

**KNOWN INEFFICIENCIES**

We noticed that the starting point for the convergence of the DFT cutoff energies was set too low after all GW calculations for all materials had been done. 
The convergence script *qe_convergence_SG15.py* therefore performed unnecessarily many DFT calculations in order to converge the total energy
with respect to the plane-wave cutoff. This does not affect the GW results in any way, but it is a waste of computational time. 
We did not modify/optimize the code for publication as this is how the data was generated. 
However, we strongly recommend that you adjust the starting values of the DFT cutoff energies when using our code.

Unfortunately, we made a mistake when converting the smearing energy from eV to Ry. Therefore, we have used a smearing of 6.25 meV instead of 25 meV in all bulk calculations. This does not affect the results in any way, since all the materials studied were semiconductors or insulators.

**2D MATERIALS**

We added 2D materials as an afterthought. Therefore, we determine that a given material is 2D by checking if the c-axis of the unit cell is greater than 15 angstroms. One could also additionally check the aspect ratio of the cells. So be very careful if you try to use this package for structures with larger cell sizes from the Materials Project. Errors or unwanted behavior will occur.

**ACKNOWLEDGEMENT**

We would like to thank Miguel A. L. Marques for providing the automated symmetry detection that supports the Quantum ESPRESSO workflows.

**LICENCE**

Copyright (c) 2023 Max Großmann and Malte Grunert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
