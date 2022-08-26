#! /usr/bin/env python

""" This code was used to produce first scan across `pce_order`. It produces
a plot of:
 - Mean +/- standard deviation
 - First Sobol indices
 - Total Sobol indices
for gamma (growth rate) and omega (frequency) at many values of `pce_order`.
"""

import time
import chaospy as cp
import numpy as np
import xarray as xr
import pickle
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from easyvvuq import Campaign
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions
from easyvvuq.sampling import PCESampler
from gs2uq import GS2Encoder, GS2Decoder

def params():
    """Define parameter space in format 
        params = {"group::param": {
                    "type": type, 
                    "min": min,
                    "max": max,
                    "default": default}
        }
    where `group` and `param` are the names of the group and parameter in the GS2 Namelist 
    See: https://gyrokinetics.gitlab.io/gs2/page/namelists/index.html 
    Only `type` and `default` are required by EasyVVUQ.
    """
    return {
        "species_parameters_1::fprim": {
            "type": "float", 
            "min": 0.8*2.2, 
            "max": 1.2*2.2, 
            "default": 2.2},
        "species_parameters_1::tprim": {
            "type": "float", 
            "min": 0.8*6.9, 
            "max": 1.2*6.9, 
            "default": 6.9}
    }

def vary():
    """Define distributions to sample over for each varying quantity in format
        vary = {
            "group::param": cp.Distribution
        }
    using ChaosPy distributions. See: https://chaospy.readthedocs.io/en/master/user_guide/index.html 
    
    Other possible quantities include:
     > list relevant quantities here
    """
    return {
        "species_parameters_1::fprim":   cp.Uniform(0.8 * 2.2, 1.2 * 2.2),  # +/- 20%
        "species_parameters_1::tprim":    cp.Uniform(0.8 * 6.9, 1.2 * 6.9)
    }
    

def run_campaign(pce_order=2, nprocs=4, gs2_bin="/home/userfs/b/bc1264/Documents/gs2/bin/gs2"):
    """Main UQ loop. Sets up the campaign, the encoder and decoder,
    and tells EasyVVUQ how to execute EasyVVUQ.

    Here I am using polynomial chaos expansion (PCE) to sample the parameter space.

    Be sure to set `nprocs` to a suitable value and `gs2_bin` to your own GS2 bin.

    For more info on UQ, see Section 2 of https://onlinelibrary.wiley.com/doi/full/10.1002/adts.201900246 
    """

    times = np.zeros(7)
    time_start = time.time()
    time_start_whole = time_start
    
    # Set up a fresh campaign called "itg_uq"
    campaign = Campaign(name="itg_uq")

    encoder = GS2Encoder(template_filename="flsm_comb.in",
                         target_filename="flsm_new.in")

    decoder = GS2Decoder(target_filename="flsm_new.out.nc")

    execute = ExecuteLocal(f"nice -n 10 mpirun -n {nprocs} {gs2_bin} flsm_new.in > GS2_print.txt",
                           stdout="GS2_print.txt")

    actions = Actions(CreateRunDirectory(root="."),
                Encode(encoder), execute, Decode(decoder))

    campaign.add_app(name="scan", params=params(), actions=actions)

    time_end = time.time()
    times[1] = time_end - time_start
    print(f"Time for phase 1 (initialising campaign) = {times[1]:.2f} s")

    # Create the sampler & associate with campaign
    time_start = time.time()
    campaign.set_sampler(PCESampler(vary=vary(), 
                         polynomial_order=pce_order))
    
    # Draw all samples (from finite set of samples)
    campaign.draw_samples()
    print(f"PCE order = {pce_order}")
    print(f"Number of samples = {campaign.get_active_sampler().count}")
    time_end = time.time()
    times[2] = time_end - time_start
    print(f"Time for phase 2 (drawing samples) = {times[2]:.2f} s")

    # Execute the campaign
    # ensure nsamples * nprocs < number of cores if sequential=False
    time_start = time.time()
    campaign.execute(sequential=True).collate(progress_bar=True)     
    time_end = time.time()
    times[3] = time_end - time_start
    print(f"Time for phase 3 (executing campaign) = {times[3]:.2f} s")

    # Get results
    time_start = time.time()
    results_df = campaign.get_collation_result()
    time_end = time.time()
    times[4] = time_end - time_start
    print(f"Time for phase 4 (getting results) = {times[4]:.2f} s")

    # Post-processing analysis
    time_start = time.time()
    results = campaign.analyse(qoi_cols=["ky", "omega/4", "gamma"])
    time_end = time.time()
    times[5] = time_end - time_start
    print(f"Time for phase 5 (post processing) = {times[5]:.2f} s")

    # Save results
    time_start = time.time()
    pickle.dump(results, open("GS2_ITG_results.pickle", "bw"))
    time_end = time.time()
    times[6] = time_end - time_start
    print(f"Time for phase 6 (saving results) = {times[6]:.2f} s")

    times[0] = time_end - time_start_whole
    print(f"Total time taken = {times[0]:.2f} s")

    return results_df, results, times, pce_order, campaign.get_active_sampler().count


if __name__ == "__main__":
    """ Run the campaign and plot results.
    """
    all_results = {1: {},
                   2: {},
                   3: {},
                   4: {},
                   5: {}}

    for pce_order in [2, 3, 4, 5]:
        R = {}

        (R["results_df"], 
        R["results"], 
        R["times"], 
        R["pce_order"], 
        R["number_of_samples"]) = run_campaign(pce_order, nprocs=16)

        # Extract results using Dimits normalisation, not GS2's
        all_results[pce_order]["ky"] = R["results"].describe("ky", "mean") / np.sqrt(2)
        all_results[pce_order]["omega"] = R["results"].describe("omega/4", "mean") * (-np.sqrt(2) / 2.2)
        all_results[pce_order]["omega_std"] = R["results"].describe("omega/4", "std") * (np.sqrt(2) / 2.2)
        all_results[pce_order]["gamma"] = R["results"].describe("gamma", "mean") * (np.sqrt(2) / 2.2)
        all_results[pce_order]["gamma_std"] = R["results"].describe("gamma", "std") * (np.sqrt(2) / 2.2)
    
        # Extract first-order and total Sobol indices
        all_results[pce_order]["sobols_first_omega"] = R["results"].sobols_first()["omega/4"]
        all_results[pce_order]["sobols_first_gamma"] = R["results"].sobols_first()["gamma"]
        all_results[pce_order]["sobols_total_omega"] = R["results"].sobols_total()["omega/4"]
        all_results[pce_order]["sobols_total_gamma"] = R["results"].sobols_total()["gamma"]

        # Extract distribution for max. growth rate
        i = np.argmax(all_results[pce_order]["omega"])
        distribution = R["results"].raw_data["output_distributions"]["gamma"].samples[i]
        all_results[pce_order]["gamma_distribution"] = distribution

    # Save results
    with open("pce_test_results.pickle", "wb") as f:
        pickle.dump(all_results, f)

    # Load results
    with open("pce_test_results.pickle", "rb") as f:
        all_results = pickle.load(f)
    del all_results[1]  # pce_order = 1 not used

    # Plot means and standard deviations
    plt.figure(1)
    colours = ["blue", "orange", "green", "purple"]
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        ky = results["ky"]
        omega = results["omega"]
        omega_std = results["omega_std"]
        gamma = results["gamma"]
        gamma_std = results["gamma_std"]

        plt.plot(ky, omega, "o-", color=colours[pce_order-2], label=f"pce_order = {pce_order}")
        plt.plot(ky, omega - omega_std, "--", color=colours[pce_order-2])
        plt.plot(ky, omega + omega_std, "--", color=colours[pce_order-2])
        plt.fill_between(ky, omega - omega_std, omega + omega_std, color=colours[pce_order-2], alpha=0.2)
        plt.plot(ky, gamma, "o-", color=colours[pce_order-2])
        plt.plot(ky, gamma - gamma_std, "--", color=colours[pce_order-2])
        plt.plot(ky, gamma + gamma_std, "--", color=colours[pce_order-2])
        plt.fill_between(ky, gamma - gamma_std, gamma + gamma_std, color=colours[pce_order-2], alpha=0.2)

    plt.legend(loc=3)
    plt.text(0.5, 0.11, r"$\gamma$", color="k", fontsize=14)
    plt.text(0.47, -0.075, r"$\omega/4$", color="k", fontsize=14)
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")

    # Plot first-order Sobols (omega)
    plt.figure(2)
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        ky = results["ky"]
        sobols_first_omega = results["sobols_first_omega"]
        for param in sobols_first_omega.keys():
            if "fprim" in param: 
                plt.plot(ky, sobols_first_omega[param], "o-", color=colours[pce_order-2], label=f"pce_order = {pce_order}")
            else:
                plt.plot(ky, sobols_first_omega[param], "o-", color=colours[pce_order-2])

    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel("First order Sobol index")
    plt.text(0.4, 0.4, "fprim", color="k", fontsize=14)
    plt.text(0.4, 0.6, "tprim", color="k", fontsize=14)
    plt.title("First order Sobol indices for " + r"$\omega_r/4$")

    # Plot first-order Sobols (gamma)
    plt.figure(3)
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        ky = results["ky"]
        sobols_first_gamma = results["sobols_first_gamma"]
        for param in sobols_first_gamma.keys():
            if "fprim" in param: 
                plt.plot(ky, sobols_first_gamma[param], "o-", color=colours[pce_order-2], label=f"pce_order = {pce_order}")
            else:
                plt.plot(ky, sobols_first_gamma[param], "o-", color=colours[pce_order-2])

    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel("First order Sobol index")
    plt.text(0.4, 0.15, "fprim", color="k", fontsize=14)
    plt.text(0.4, 0.85, "tprim", color="k", fontsize=14)
    plt.title("First order Sobol indices for " + r"$\gamma$")

    # Plot total Sobols (omega)
    plt.figure(4)
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        ky = results["ky"]
        sobols_total_omega = results["sobols_total_omega"]
        for param in sobols_total_omega.keys():
            if "fprim" in param: 
                plt.plot(ky, sobols_total_omega[param], "o-", color=colours[pce_order-2], label=f"pce_order = {pce_order}")
            else:
                plt.plot(ky, sobols_total_omega[param], "o-", color=colours[pce_order-2])

    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel("Total Sobol index")
    plt.text(0.4, 0.4, "fprim", color="k", fontsize=14)
    plt.text(0.4, 0.6, "tprim", color="k", fontsize=14)
    plt.title("Total Sobol indices for " + r"$\omega_r/4$")

    # Plot total Sobols (gamma)
    plt.figure(5)
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        ky = results["ky"]
        sobols_total_gamma = results["sobols_total_gamma"]
        for param in sobols_total_gamma.keys():
            if "fprim" in param: 
                plt.plot(ky, sobols_total_gamma[param], "o-", color=colours[pce_order-2], label=f"pce_order = {pce_order}")
            else:
                plt.plot(ky, sobols_total_gamma[param], "o-", color=colours[pce_order-2])

    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel("Total Sobol index")
    plt.text(0.4, 0.15, "fprim", color="k", fontsize=14)
    plt.text(0.4, 0.85, "tprim", color="k", fontsize=14)
    plt.title("Total Sobol indices for " + r"$\gamma$")

    # Plot distributions (max. gamma)
    plt.figure(6)
    for pce_order in all_results.keys():
        results = all_results[pce_order]
        gamma = results["gamma"]
        gamma_distribution = results["gamma_distribution"]

        i = np.argmax(gamma)
        mean_gamma = gamma[i]

        pdf_kde_samples = cp.GaussianKDE(gamma_distribution)
        _gamma = np.linspace(pdf_kde_samples.lower, pdf_kde_samples.upper[0], 101)
        plt.semilogy(_gamma, pdf_kde_samples.pdf(_gamma), label=f"pce_order = {pce_order}", color=colours[pce_order-2])
        plt.semilogy(mean_gamma, pdf_kde_samples.pdf(mean_gamma), "bo")

    plt.legend()
    plt.xlabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")
    plt.ylabel("Distribution function")
    plt.title("Distribution function for max. " + r"$\gamma$")

    plt.show()
