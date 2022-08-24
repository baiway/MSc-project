""" The results from the EasyVVUQ analysis are saved as .pickle files
if the run completes successfully. These can be opened using the code snippet below
"""

import numpy as np


results = open("GS2_ITG_results.pickle", "rb")

# Extract results using GS2 normalisation
ky = results.describe("ky", "mean") / np.sqrt(2)
omega = results.describe("omega/4", "mean") * (-np.sqrt(2) / 2.2)
omega_std = results.describe("omega/4", "std") * (np.sqrt(2) / 2.2)
gamma = results.describe("gamma", "mean") * (np.sqrt(2) / 2.2)
gamma_std = results.describe("gamma", "std") * (np.sqrt(2) / 2.2)