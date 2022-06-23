#! /usr/bin/env python

import os
import easyvvuq as uq
import chaospy as cp
import pickle # what does this do?
import time
import numpy as np
import matplotlib.pylab as plt

# Set up a fresh campaign called "fusion_pce."
my_campaign = uq.Campaign(name='fusion_pce.')

# Define parameter space
params = {
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
    "out_file": {"type": "string",  "default": "output.csv"}
}

# Create an encoder, decoder and collater for PCE test app
encoder = uq.encoders.GenericEncoder(template_fname='fusion.template',
                                     delimiter='$',
                                     target_filename='fusion_in.json')


decoder = uq.decoders.SimpleCSV(target_filename="output.csv",
                                output_columns=["te", "ne", "rho", "rho_norm"])

#collater = uq.collate.AggregateSamples(average=False)

# Add the app (automatically set as current app)
my_campaign.add_app(name="fusion",
                    params=params,
                    encoder=encoder,
                    decoder=decoder)

# Create the sampler (here simplified to two uncertain quantities)
vary = {
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

# Associate a sampler with the campaign
my_campaign.set_sampler(uq.sampling.PCESampler(vary=vary, polynomial_order=3))

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
print('Number of samples = %s' % my_campaign.get_active_sampler().count)

# Create and populate the run directories
my_campaign.populate_runs_dir()

# Run the cases
cwd = os.getcwd().replace(' ', '\ ')    # deals with possible spaces in path
cmd = f"{cwd}/fusion_model.py fusion_in.json"
print(cmd)
my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(cmd, interpret='python3'))

# Collate the results
my_campaign.collate()

# Post-processing analysis
my_campaign.apply_analysis(uq.analysis.PCEAnalysis(sampler=my_campaign.get_active_sampler(),
                                                   qoi_cols=["te", "ne", "rho", "rho_norm"]))

# Get Descriptive Statistics
results = my_campaign.get_last_analysis()
rho = results.describe('rho', 'mean')
rho_norm = results.describe('rho_norm', 'mean')

# plot the calculated Te: mean, with std deviation, 10 and 90% and range
plt.figure()
plt.plot(rho, results.describe('te', 'mean'), 'b-', label='Mean')
plt.plot(rho, results.describe('te', 'mean')-results.describe('te', 'std'), 'b--', label='+1 std deviation')
plt.plot(rho, results.describe('te', 'mean')+results.describe('te', 'std'), 'b--')
plt.fill_between(rho, results.describe('te', 'mean')-results.describe('te', 'std'), results.describe('te', 'mean')+results.describe('te', 'std'), color='b', alpha=0.2)
plt.plot(rho, results.describe('te', '10%'), 'b:', label='10 and 90 percentiles')
plt.plot(rho, results.describe('te', '90%'), 'b:')
plt.fill_between(rho, results.describe('te', '10%'), results.describe('te', '90%'), color='b', alpha=0.1)
plt.fill_between(rho, results.describe('te', 'min'), results.describe('te', 'max'), color='b', alpha=0.05)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('Te [eV]')
plt.title(my_campaign.campaign_dir);

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