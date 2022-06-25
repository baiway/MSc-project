from pathlib import Path
import time
import easyvvuq as uq
import chaospy as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions, ExecutePython

# define parameter space
# symbols defined here: https://openturns.github.io/openturns/latest/usecases/use_case_deflection_tube.html 
def define_params():
    return {
        "F":       {"type": "float", "default": 1.0}, 
        "L":       {"type": "float", "default": 1.5}, 
        "a":       {"type": "float", "min": 0.7, "max": 1.2, "default": 1.0}, 
        "D":       {"type": "float", "min": 0.75, "max": 0.85, "default": 0.8},
        "d":       {"type": "float", "default": 0.1},
        "E":       {"type": "float", "default": 200000},
        "outfile": {"type": "string", "default": "output.json"}

    }

# define varying space to sample over
def define_vary():
    return {
        "F":   cp.Normal(1, 0.1),
        "L":   cp.Normal(1.5, 0.01),
        "a":   cp.Uniform(0.7, 1.2),
        "D":   cp.Triangle(0.75, 0.8, 0.85)
    }


def run_beam_model(input):
    import beam
    qois = ['g1']
    del input['outfile']
    return {qois[0]: beam.evaluate_g1(**input)}

def run_campaign(pce_order=2, use_files=False):

    times = np.zeros(7)
    time_start = time.time()
    time_start_whole = time_start

    # set up a fesh campaign called "beam_pce"
    campaign = uq.Campaign(name='beam_pce')

    if use_files:
        encoder = uq.encoders.GenericEncoder(template_fname='beam.template', 
                                             delimiter='$', 
                                             target_filename='input.json')

        decoder = uq.decoders.JSONDecoder(target_filename='output.json', 
                                          output_columns=['g1'])
        
        cwd = Path().cwd().as_posix() # makes path names play nice across operating systems
        execute = ExecuteLocal("python {}/beam.py input.json".format(cwd))

        actions = Actions(CreateRunDirectory('/tmp'),
                    Encode(encoder), execute, Decode(decoder))
    
    else:
        actions = Actions(ExecutePython(run_beam_model))
    
    campaign.add_app(name='beam', params=define_params(), actions=actions)

    time_end = time.time()
    times[1] = time_end - time_start
    print("Time for phase 1 (initialising campaign) = {0:.2f} s".format(times[1]))

    # create the sampler & associate with campaign
    time_start = time.time()
    campaign.set_sampler(uq.sampling.PCESampler(vary=define_vary(), 
                                                polynomial_order=pce_order))
    # draw all samples (from finite set of samples)
    campaign.draw_samples()
    print("PCE order = {}".format(pce_order))
    print("Number of samples = {}".format(campaign.get_active_sampler().count))
    time_end = time.time()
    times[2] = time_end - time_start
    print("Time for phase 2 (drawing samples) = {0:.2f} s".format(times[2]))

    # Execute the campaign
    time_start = time.time()
    campaign.execute(sequential=True).collate(progress_bar=True)
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
    results = campaign.analyse(qoi_cols=["g1"])
    time_end = time.time()
    times[5] = time_end - time_start
    print("Time for phase 5 (post processing) = {0:.2f} s".format(times[5]))

    # Save results
    time_start = time.time()
    pickle.dump(results, open('beam_results.pickle', 'bw'))
    time_end = time.time()
    times[6] = time_end - time_start
    print("Time for phase 6 (saving results) = {0:.2f} s".format(times[6]))

    times[0] = time_end - time_start_whole
    print("Total time taken = {0:.2f} s".format(times[0]))

    return results_df, results, times, pce_order, campaign.get_active_sampler().count

if __name__ == '__main__':
    R = {}
    for pce_order in range(1, 5):
        R[pce_order] = {}
        (R[pce_order]['results_df'], 
        R[pce_order]['results'], 
        R[pce_order]['times'], 
        R[pce_order]['order'], 
        R[pce_order]['number_of_samples']) = run_campaign(pce_order=pce_order, use_files=False)