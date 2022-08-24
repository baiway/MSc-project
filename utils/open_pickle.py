""" The results from the EasyVVUQ analysis are saved as .pickle files
if the run completes successfully. These can be opened using the code snippet below

NOT CURRENTLY FUNCTIONAL
"""

import numpy as np
import matplotlib.pyplot as plt

results = open("GS2_ITG_results.pickle", "rb")

# Extract results using GS2 normalisation
ky = results.describe("ky", "mean") / np.sqrt(2)
omega = results.describe("omega/4", "mean") * (-np.sqrt(2) / 2.2)
omega_std = results.describe("omega/4", "std") * (np.sqrt(2) / 2.2)
gamma = results.describe("gamma", "mean") * (np.sqrt(2) / 2.2)
gamma_std = results.describe("gamma", "std") * (np.sqrt(2) / 2.2)

# Plot the mean frequency and growth rate and std deviations
plt.figure(1)
plt.plot(ky, omega, "o-", color="orange", label=r"$\overline{\omega}_r/4 \pm \sigma_{\omega_r/4}$")
plt.plot(ky, omega - omega_std, "--", color="orange")
plt.plot(ky, omega + omega_std, "--", color="orange")
plt.fill_between(ky, omega - omega_std, omega + omega_std, color="orange", alpha=0.4)
plt.plot(ky, gamma, "o-", color="blue", label=r"$\overline{\gamma} \pm \sigma_\gamma$")
plt.plot(ky, gamma - gamma_std, "--", color="blue")
plt.plot(ky, gamma + gamma_std, "--", color="blue")
plt.fill_between(ky, gamma - gamma_std, gamma + gamma_std, color="blue", alpha=0.2)
plt.legend()
plt.xlabel(r"$k_y\rho$")
plt.ylabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")

plt.savefig("freq_and_growth_rate_stds.png", dpi=300)
plt.show()
plt.clf()