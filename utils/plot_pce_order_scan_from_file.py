import pickle
import numpy as np
import matplotlib.pyplot as plt

all_results = {2: {},
               3: {},
               4: {},
               5: {},
               6: {},
               7: {}}

pce_orders = [2, 3, 4, 5]

for pce_order in pce_orders:
    with open(f"processed_results_pce_order_{pce_order}.pickle", "rb") as f:
        all_results[pce_order] = pickle.load(f)

xticks = [f"PO={p}" + "\n" f"runs={(p+1)**2}" for p in pce_orders[:-1]]

omega_err, omega_std_err, gamma_err, gamma_std_err = [], [], [], []
first_sobols_omega_fprim_err, first_sobols_omega_tprim_err = [], []
first_sobols_gamma_fprim_err, first_sobols_gamma_tprim_err = [], []

li = pce_orders[-1] # last index

for pce_order in pce_orders[:-1]:
    omega_err.append(np.mean(np.abs(all_results[pce_order]["omega"] - all_results[li]["omega"]) / all_results[li]["omega"]))
    omega_std_err.append(np.mean(np.abs(all_results[pce_order]["omega_std"] - all_results[li]["omega_std"]) / all_results[li]["omega_std"]))
    gamma_err.append(np.mean(np.abs(all_results[pce_order]["gamma"] - all_results[li]["gamma"]) / all_results[li]["gamma"]))
    gamma_std_err.append(np.mean(np.abs(all_results[pce_order]["gamma_std"] - all_results[li]["gamma_std"]) / all_results[li]["gamma_std"]))

    first_sobols_omega_fprim_err.append(np.mean(np.abs(all_results[pce_order]["sobols_first_omega"]["species_parameters_1::fprim"] - all_results[li]["sobols_first_omega"]["species_parameters_1::fprim"]) / all_results[li]["sobols_first_omega"]["species_parameters_1::fprim"]))
    first_sobols_omega_tprim_err.append(np.mean(np.abs(all_results[pce_order]["sobols_first_omega"]["species_parameters_1::tprim"] - all_results[li]["sobols_first_omega"]["species_parameters_1::tprim"]) / all_results[li]["sobols_first_omega"]["species_parameters_1::tprim"]))
    first_sobols_gamma_fprim_err.append(np.mean(np.abs(all_results[pce_order]["sobols_first_gamma"]["species_parameters_1::fprim"] - all_results[li]["sobols_first_gamma"]["species_parameters_1::fprim"]) / all_results[li]["sobols_first_gamma"]["species_parameters_1::fprim"]))
    first_sobols_gamma_tprim_err.append(np.mean(np.abs(all_results[pce_order]["sobols_first_gamma"]["species_parameters_1::tprim"] - all_results[li]["sobols_first_gamma"]["species_parameters_1::tprim"]) / all_results[li]["sobols_first_gamma"]["species_parameters_1::tprim"]))

# Plot omega errors
plt.figure(1)
plt.semilogy(pce_orders[:-1], omega_err, "o-", label="mean")
plt.semilogy(pce_orders[:-1], omega_std_err, "o-", label="std")
plt.xticks(ticks=pce_orders[:-1], labels=xticks, rotation=90)
plt.xlabel("Polynomial order")
plt.ylabel(f"Relative error compared to pce_order = {pce_orders[-1]}")
plt.title("Relative error in " + r"$\omega_r/4$" + " averaged across " + r"$k_y\rho$")
plt.legend()
plt.tight_layout()
plt.show()

# Plot gamma errors
plt.figure(2)
plt.semilogy(pce_orders[:-1], gamma_err, "o-", label="mean")
plt.semilogy(pce_orders[:-1], gamma_std_err, "o-", label="std")
plt.xticks(ticks=pce_orders[:-1], labels=xticks, rotation=90)
plt.xlabel("Polynomial order")
plt.ylabel(f"Relative error compared to pce_order = {pce_orders[-1]}")
plt.title("Relative error in " + r"$\gamma$" + " averaged across " + r"$k_y\rho$")
plt.legend()
plt.tight_layout()
plt.show()

# Plot first Sobol errors in omega
plt.figure(3)
plt.semilogy(pce_orders[:-1], first_sobols_omega_fprim_err, "o-", label="fprim")
plt.semilogy(pce_orders[:-1], first_sobols_omega_tprim_err, "o-", label="tprim")
plt.xticks(ticks=pce_orders[:-1], labels=xticks, rotation=90)
plt.xlabel("Polynomial order")
plt.ylabel(f"Relative error compared to pce_order = {pce_orders[-1]}")
plt.title("Relative error in first Sobol index for " + r"$\omega_r/4$" + " averaged across " + r"$k_y\rho$")
plt.legend()
plt.tight_layout()
plt.show()

# Plot first Sobol errors in gamma
plt.figure(4)
plt.semilogy(pce_orders[:-1], first_sobols_gamma_fprim_err, "o-", label="fprim")
plt.semilogy(pce_orders[:-1], first_sobols_gamma_tprim_err, "o-", label="tprim")
plt.xticks(ticks=pce_orders[:-1], labels=xticks, rotation=90)
plt.xlabel("Polynomial order")
plt.ylabel(f"Relative error compared to pce_order = {pce_orders[-1]}")
plt.title("Relative error in first Sobol index for " + r"$\gamma$" + " averaged across " + r"$k_y\rho$")
plt.legend()
plt.tight_layout()
plt.show()
