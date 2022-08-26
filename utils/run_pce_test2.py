#! /usr/bin/env python

""" This code was used in the refined scan across `pce_order`. It produces
a plot of the first Sobol index for omega (frequency). Due to the high number
of runs at `pce_order` > 5, it's recommended to just run this for one or two
mode numbers.
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
    all_results = {2: {},
                   3: {},
                   4: {},
                   5: {},
                   6: {},
                   7: {}}

    pce_orders = [2, 3, 4, 5, 6, 7]
    for pce_order in pce_orders:
        R = {}

        (R["results_df"], 
        R["results"], 
        R["times"], 
        R["pce_order"], 
        R["number_of_samples"]) = run_campaign(pce_order, nprocs=16)
    
        # Extract first-order Sobol indices for omega
        all_results[pce_order]["sobols_first_omega"] = R["results"].sobols_first()["omega/4"]

    # Save results
    with open("pce_test_results2.pickle", "wb") as f:
        pickle.dump(all_results, f)

    # Load results
    with open("pce_test_results2.pickle", "rb") as f:
        all_results = pickle.load(f)

    # Plot first-order Sobols (omega)
    fprim_sobol1, fprim_sobol2, tprim_sobol1, tprim_sobol2 = [], [], [], []
    for pce_order in all_results.keys():
        results = all_results[pce_order]

        fprim_sobol1.append(results["sobols_first_omega"]["species_parameters_1::fprim"][0])
        fprim_sobol2.append(results["sobols_first_omega"]["species_parameters_1::fprim"][1])
        tprim_sobol1.append(results["sobols_first_omega"]["species_parameters_1::tprim"][0])
        tprim_sobol2.append(results["sobols_first_omega"]["species_parameters_1::tprim"][1])

    fprim_sobol1_err = np.abs(np.array(fprim_sobol1) - fprim_sobol1[-1])
    fprim_sobol2_err = np.abs(np.array(fprim_sobol2) - fprim_sobol2[-1])
    tprim_sobol1_err = np.abs(np.array(tprim_sobol1) - tprim_sobol1[-1])
    tprim_sobol2_err = np.abs(np.array(tprim_sobol2) - tprim_sobol2[-1])

    xticks = [f"PO={p}" + "\n" f"runs={(p+1)**2}" for p in pce_orders[:-1]]

    plt.figure(1)
    plt.semilogy(pce_orders[:-1], fprim_sobol1_err[:-1], "o-", label="fprim")
    plt.semilogy(pce_orders[:-1], tprim_sobol1_err[:-1], "o-", label="tprim")
    plt.xticks(ticks = pce_orders[:-1], labels=xticks, rotation=90)
    plt.xlabel("Polynomial order")
    plt.ylabel("Error in first Sobol index compared to order=7")
    plt.title("Error in first order Sobol index for " + r"$\omega_r/4$" + " at " + r"$k_y\rho=0.05$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.semilogy(pce_orders[:-1], fprim_sobol2_err[:-1], "o-", label="fprim")
    plt.semilogy(pce_orders[:-1], tprim_sobol2_err[:-1], "o-", label="tprim")
    plt.xticks(ticks = pce_orders[:-1], labels=xticks, rotation=90)
    plt.xlabel("Polynomial order")
    plt.ylabel("Error in first Sobol index compared to order=7")
    plt.title("Error in first order Sobol index for " + r"$\omega_r/4$" + " at " + r"$k_y\rho=0.50$")
    plt.legend()
    plt.tight_layout()
    plt.show()


