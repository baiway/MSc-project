#! /usr/bin/env python

import time
import chaospy as cp
import numpy as np
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
    (see: https://gyrokinetics.gitlab.io/gs2/page/namelists/index.html). Only `type` and `default`
    are required. 
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
    """Main UQ loop
     > add to this
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
    time_start = time.time()
    campaign.execute(sequential=True).collate(progress_bar=True)     # ensure nsamples * nprocs < number of cores
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
    R = {}

    (R["results_df"], 
    R["results"], 
    R["times"], 
    R["pce_order"], 
    R["number_of_samples"]) = run_campaign(pce_order=2, nprocs=8)

    # I think (?) EasyVVUQ returns these as np.arrays, not lists
    ky_dim = R["results"].describe("ky", "mean") / np.sqrt(2)
    omega = R["results"].describe("omega/4", "mean") * (-np.sqrt(2) / 2.2)
    omega_std = R["results"].describe("omega/4", "std") * (np.sqrt(2) / 2.2)
    gamma = R["results"].describe("gamma", "mean") * (np.sqrt(2) / 2.2)
    gamma_std = R['results'].describe("gamma", "std") * (np.sqrt(2) / 2.2)

    # Plot the calculated rates: mean with std deviation
    plt.figure()
    plt.plot(ky_dim, omega, "o-", color="orange", label=r"$\overline{\omega}_r/4 \pm \sigma_{\omega_r/4}$")
    plt.plot(ky_dim, omega - omega_std, "--", color="orange")
    plt.plot(ky_dim, omega + omega_std, "--", color="orange")
    plt.fill_between(ky_dim, omega - omega_std, omega + omega_std, color="orange", alpha=0.4)
    plt.plot(ky_dim, gamma, "o-", color="blue", label=r"$\overline{\gamma} \pm \sigma_\gamma$")
    plt.plot(ky_dim, gamma - gamma_std, "--", color="blue")
    plt.plot(ky_dim, gamma + gamma_std, "--", color="blue")
    plt.fill_between(ky_dim, gamma - gamma_std, gamma + gamma_std, color="blue", alpha=0.2)
    """
    plt.plot(rho, Te_10, 'b:', label='10 and 90 percentiles')
    plt.plot(rho, Te_90, 'b:')
    plt.fill_between(rho, Te_10, Te_90, color='b', alpha=0.1)
    plt.fill_between(rho, Te_min, Te_max, color='b', alpha=0.05)
    """
    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")

    plt.savefig("freq_and_growth_rate_stds.png", dpi=300)
    plt.show()
    plt.clf()

    # Plot the first Sobol results
    #plt.figure()
    #for k in results.sobols_first()['te'].keys(): plt.plot(rho, results.sobols_first()['te'][k], label=k)

"""
# plot the first Sobol results
plt.figure()
for k in results.sobols_first()['te'].keys(): plt.plot(rho, results.sobols_first()['te'][k], label=k)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('sobols_first')
plt.title(my_campaign.campaign_dir);

# plot the second Sobol results
plt.figure()
for k1 in results.sobols_second()['te'].keys():
    for k2 in results.sobols_second()['te'][k1].keys():
        plt.plot(rho, results.sobols_second()['te'][k1][k2], label=k1+'/'+k2)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('sobols_second')
plt.title(my_campaign.campaign_dir+'\n');

# plot the total Sobol results
plt.figure()
for k in results.sobols_total()['te'].keys(): plt.plot(rho, results.sobols_total()['te'][k], label=k)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('sobols_total')
plt.title(my_campaign.campaign_dir);

# plot the distributions
plt.figure()
for i, D in enumerate(results.raw_data['output_distributions']['te']):
    _Te = np.linspace(D.lower[0], D.upper[0], 101)
    _DF = D.pdf(_Te)
    plt.loglog(_Te, _DF, 'b-', alpha=0.25)
    plt.loglog(results.describe('te', 'mean')[i], np.interp(results.describe('te', 'mean')[i], _Te, _DF), 'bo')
    plt.loglog(results.describe('te', 'mean')[i]-results.describe('te', 'std')[i], np.interp(results.describe('te', 'mean')[i]-results.describe('te', 'std')[i], _Te, _DF), 'b*')
    plt.loglog(results.describe('te', 'mean')[i]+results.describe('te', 'std')[i], np.interp(results.describe('te', 'mean')[i]+results.describe('te', 'std')[i], _Te, _DF), 'b*')
    plt.loglog(results.describe('te', '10%')[i],  np.interp(results.describe('te', '10%')[i], _Te, _DF), 'b+')
    plt.loglog(results.describe('te', '90%')[i],  np.interp(results.describe('te', '90%')[i], _Te, _DF), 'b+')
plt.xlabel('Te')
plt.ylabel('distribution function');
"""
