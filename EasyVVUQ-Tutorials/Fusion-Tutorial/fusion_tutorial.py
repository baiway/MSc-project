#! /usr/bin/env python

from pathlib import Path
import time
import easyvvuq as uq
import chaospy as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions, ExecutePython


# Define parameter space
def define_params():
    return {
        "Qe_tot":   {"type": "float",   "min": 1.0e6, "max": 50.0e6, "default": 2e6},
        "H0":       {"type": "float",   "min": 0.00,  "max": 1.0,    "default": 0},
        "Hw":       {"type": "float",   "min": 0.01,  "max": 100.0,  "default": 0.1},
        "Te_bc":    {"type": "float",   "min": 10.0,  "max": 1000.0, "default": 100},
        "chi":      {"type": "float",   "min": 0.01,  "max": 100.0,  "default": 1},
        "a0":       {"type": "float",   "min": 0.2,   "max": 10.0,   "default": 1},
        "R0":       {"type": "float",   "min": 0.5,   "max": 20.0,   "default": 3},
        "E0":       {"type": "float",   "min": 1.0,   "max": 10.0,   "default": 1.5},
        "b_pos":    {"type": "float",   "min": 0.95,  "max": 0.99,   "default": 0.98},
        "b_height": {"type": "float",   "min": 3e19,  "max": 10e19,  "default": 6e19},
        "b_sol":    {"type": "float",   "min": 2e18,  "max": 3e19,   "default": 2e19},
        "b_width":  {"type": "float",   "min": 0.005, "max": 0.025,  "default": 0.01},
        "b_slope":  {"type": "float",   "min": 0.0,   "max": 0.05,   "default": 0.01},
        "nr":       {"type": "integer", "min": 10,    "max": 1000,   "default": 100},
        "dt":       {"type": "float",   "min": 1e-3,  "max": 1e3,    "default": 100},
        "outfile": {"type": "string",  "default": "output.csv"}
    }


# Define distributions to sample over for each varying quantity (here just 2)
def define_vary():
    return {
        "Qe_tot":   cp.Uniform(1.8e6, 2.2e6),
        "Te_bc":    cp.Uniform(80.0,  120.0)
    }
"""
other possible quantities include:
    "a0":       cp.Uniform(0.9,   1.1),
    "R0":       cp.Uniform(2.7,   3.3),
    "E0":       cp.Uniform(1.4,   1.6),
    "b_pos":    cp.Uniform(0.95,  0.99),
    "b_height": cp.Uniform(5e19,  7e19),
    "b_sol":    cp.Uniform(1e19,  3e19),
    "b_width":  cp.Uniform(0.015, 0.025),
    "b_slope":  cp.Uniform(0.005, 0.020)
"""


# Run fusion.py
def run_fusion_model(input):
    import fusion
    qois = ["te", "ne", "rho", "rho_norm"]
    del input['outfile']

    R = {}
    (R["te"],
    R["ne"],
    R["rho"],
    R["rho_norm"]) = fusion.solve_Te(**input)

    return R


# Main UQ campaign loop
def run_campaign(pce_order=2, use_files=False):

    times = np.zeros(7)
    time_start = time.time()
    time_start_whole = time_start
    
    # Set up a fresh campaign called "fusion_pce"
    campaign = uq.Campaign(name='fusion_pce')

    if use_files:
        encoder = uq.encoders.GenericEncoder(template_fname='fusion.template', 
                                             delimiter='$', 
                                             target_filename='input.json')

        decoder = uq.decoders.SimpleCSV(target_filename='output.csv', 
                                          output_columns=["te", "ne", "rho", "rho_norm"])
        
        cwd = Path().cwd().as_posix() # makes path names play nice across operating systems
        execute = ExecuteLocal("python3 {}/fusion_model.py input.json".format(cwd))

        actions = Actions(CreateRunDirectory('/tmp'),
                    Encode(encoder), execute, Decode(decoder))
    else:
        actions = Actions(ExecutePython(run_fusion_model))

    campaign.add_app(name='fusion', params=define_params(), actions=actions)

    time_end = time.time()
    times[1] = time_end - time_start
    print("Time for phase 1 (initialising campaign) = {0:.2f} s".format(times[1]))

    # Create the sampler & associate with campaign
    time_start = time.time()
    campaign.set_sampler(uq.sampling.PCESampler(vary=define_vary(), 
                                                polynomial_order=pce_order))
    
    # Draw all samples (from finite set of samples)
    campaign.draw_samples()
    print("PCE order = {}".format(pce_order))
    print("Number of samples = {}".format(campaign.get_active_sampler().count))
    time_end = time.time()
    times[2] = time_end - time_start
    print("Time for phase 2 (drawing samples) = {0:.2f} s".format(times[2]))

    # Execute the campaign
    time_start = time.time()
    campaign.execute().collate(progress_bar=True)
    time_end = time.time()
    times[3] = time_end - time_start
    print("Time for phase 3 (executing campaign) = {0:.2f} s".format(times[3]))

    # Get results
    time_start = time.time()
    results_df = campaign.get_collation_result()
    time_end = time.time()
    times[4] = time_end - time_start
    print("Time for phase 4 (getting results) = {0:.2f} s".format(times[4]))

    # Post-processing analysis
    time_start = time.time()
    results = campaign.analyse(qoi_cols=["te", "ne", "rho", "rho_norm"])
    time_end = time.time()
    times[5] = time_end - time_start
    print("Time for phase 5 (post processing) = {0:.2f} s".format(times[5]))

    # Save results
    time_start = time.time()
    pickle.dump(results, open('fusion_results.pickle', 'bw'))
    time_end = time.time()
    times[6] = time_end - time_start
    print("Time for phase 6 (saving results) = {0:.2f} s".format(times[6]))

    times[0] = time_end - time_start_whole
    print("Total time taken = {0:.2f} s".format(times[0]))

    return results_df, results, times, pce_order, campaign.get_active_sampler().count

if __name__ == '__main__':
    R = {}
    pce_order = 3

    (R['results_df'], 
    R['results'], 
    R['times'], 
    R['order'], 
    R['number_of_samples']) = run_campaign(pce_order=pce_order, use_files=True)

    # Get descriptive statistics
    rho = R['results'].describe('rho', 'mean')
    rho_norm = R['results'].describe('rho_norm', 'mean')
    Te = R['results'].describe('te', 'mean')
    Te_std = R['results'].describe('te', 'std')
    Te_10 = R['results'].describe('te', '10%')
    Te_90 = R['results'].describe('te', '90%')
    Te_min = R['results'].describe('te', 'min')
    Te_max = R['results'].describe('te', 'max')

    # Plot the calculated Te: mean, with std deviation, 10 and 90% and range
    plt.figure()
    plt.plot(rho, Te, 'b-', label='Mean')
    plt.plot(rho, Te - Te_std, 'b--', label='+1 std deviation')
    plt.plot(rho, Te + Te_std, 'b--')
    plt.fill_between(rho, Te - Te_std, Te + Te_std, color='b', alpha=0.2)
    plt.plot(rho, Te_10, 'b:', label='10 and 90 percentiles')
    plt.plot(rho, Te_90, 'b:')
    plt.fill_between(rho, Te_10, Te_90, color='b', alpha=0.1)
    plt.fill_between(rho, Te_min, Te_max, color='b', alpha=0.05)
    plt.legend(loc=0)
    plt.xlabel('rho [m]')
    plt.ylabel('Te [eV]')

    plt.show()

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