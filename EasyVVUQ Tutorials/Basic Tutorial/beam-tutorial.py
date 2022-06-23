import os
import easyvvuq as uq
import chaospy as cp
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions

# symbols defined here: https://openturns.github.io/openturns/latest/usecases/use_case_deflection_tube.html 
params = {
    "F": {"type": "float", "default": 1.0}, 
    "L": {"type": "float", "default": 1.5}, 
    "a": {"type": "float", "min": 0.7, "max": 1.2, "default": 1.0}, 
    "D": {"type": "float", "min": 0.75, "max": 0.85, "default": 0.8},
    "d": {"type": "float", "default": 0.1},
    "E": {"type": "float", "default": 200000},
    "outfile": {"type": "string", "default": "output.json"}
}

# beam.template is a template input file: {"outfile": "$outfile", "F": $F, "L": $L, "a": $a, "D": $D, "d": $d, "E": $E}
# the values for each key are tags (signified by '$' delimiter) which will be substituted by EasyVVUQ to sample the parameter space
encoder = uq.encoders.GenericEncoder(template_fname='beam.template', delimiter='$', target_filename='input.json')
decoder = uq.decoders.JSONDecoder(target_filename='output.json', output_columns=['g1'])

cwd = os.getcwd().replace(' ', '\ ')    # deals with possible spaces in path
input("Press Enter to continue...")

execute = ExecuteLocal("{}/beam input.json".format(cwd))

actions = Actions(CreateRunDirectory('/tmp'),
                    Encode(encoder), execute, Decode(decoder))

campaign = uq.Campaign(name='beam', params=params, actions=actions)

# define input parameter distributions
vary = {
    "F": cp.Normal(1, 0.1),
    "L": cp.Normal(1.5, 0.01),
    "a": cp.Uniform(0.7, 1.2),
    "D": cp.Triangle(0.75, 0.8, 0.85)
}

# use polynomial chaos expansion (PCE) sampler
campaign.set_sampler(uq.sampling.PCESampler(vary=vary, polynomial_order=1))

# execute the campaign
campaign.execute().collate()
