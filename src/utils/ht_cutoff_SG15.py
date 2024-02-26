"""
DO NOT USE THIS FOR FUTURE CALCULATIONS. THIS IS ONLY FOR REPRODUCING THE RESULTS OF THE PAPER.
THE CUTOFF STARTING POINTS ARE TOO LOW FOR THE SG15 PSEUDOPOTENTIALS AND THE DFT CONVERGENCE WORKFLOW.
THEREFORE, UNNECESSARILY MANY CALCULATIONS ARE PERFORMED TO CONVERGE THE TOTAL ENERGY WITH RESPECT TO THE PLANE WAVE CUTOFF.
"""
data = {
    "H": 36,
    "He": 45,
    "Li": 37,
    "Be": 44,
    "B": 38,
    "C": 41,
    "N": 42,
    "O": 42,
    "F": 42,
    "Ne": 34,
    "Na": 44,
    "Mg": 42,
    "Al": 20,
    "Si": 18,
    "P": 22,
    "S": 26,
    "Cl": 29,
    "Ar": 33,
    "K": 37,
    "Ca": 34,
    "Sc": 39,
    "Ti": 42,
    "V": 42,
    "Cr": 47,
    "Mn": 48,
    "Fe": 45,
    "Co": 48,
    "Ni": 49,
    "Cu": 46,
    "Zn": 42,
    "Ga": 40,
    "Ge": 39,
    "As": 42,
    "Se": 43,
    "Br": 23,
    "Kr": 34,
    "Rb": 23,
    "Sr": 34,
    "Y": 36,
    "Zr": 33,
    "Nb": 41,
    "Mo": 40,
    "Tc": 42,
    "Ru": 42,
    "Rh": 44,
    "Pd": 41,
    "Ag": 41,
    "Cd": 51,
    "In": 35,
    "Sn": 36,
    "Sb": 40,
    "Te": 40,
    "I": 35,
    "Xe": 34,
    "Cs": 25,
    "Ba": 22,  # skipping the lanthanoide
    "Hf": 29,
    "Ta": 29,
    "W": 37,
    "Re": 36,
    "Os": 37,
    "Ir": 34,
    "Pt": 42,
    "Au": 38,
    "Hg": 33,
    "Tl": 31,
    "Pb": 28,
    "Bi": 33,
    "Po": 32,
    "At": 40,  # guess ...
    "Rn": 36,  # ignoring all other elements at the moment...
}