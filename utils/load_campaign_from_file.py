from easyvvuq import Campaign
from easyvvuq.sampling import PCESampler
from pathlib import Path
import xarray as xr
import f90nml
import numpy as np
import chaospy as cp
import pickle

class GS2OutputDecoder:
    def __init__(self, target_filename):
        self.target_filename = target_filename
    
    def parse_sim_output(self, run_info={}) -> dict:
        """Parses the GS2 output file (NetCDF) and converts it to the EasyVVUQ internal 
        dictionary based format. The output has the form 
         {"aky": [...], "omega/4": [...], "gamma": [...]}.
        Parameters
        ----------
        run_info: dict
            Information about the run (used to retrieve construct the absolute path
            to the NetCDF file that needs decoding.
        """

        results = {"ky": None, "omega/4": None, "gamma": None}

        run_dir = Path(run_info['run_dir'])
        output_filepath = run_dir / self.target_filename

        # Save results as lists of values rather than NumPy arrays (required by EasyVVUQ)
        with xr.open_dataset(output_filepath, engine="netcdf4") as ds:
            results["ky"] = ds.ky.data.tolist()
            results["omega/4"] = (ds.omega_average.isel(ri=0, t=-1).squeeze().data / 4).tolist()
            results["gamma"] = ds.omega_average.isel(ri=1, t=-1).squeeze().data.tolist()
        
        return results

class GS2InputDecoder:
    def __init__(self, target_filename):
        self.target_filename = target_filename
    
    def parse_sim_output(self, run_info={}) -> dict:
        results = {"fprim": None, "tprim": None}

        run_dir = Path(run_info['run_dir'])
        input_filepath = run_dir / self.target_filename

        # Open template filename
        with open(input_filepath, "r") as f:
            nml = f90nml.read(f)

        results["fprim"] = nml["species_parameters_1"]["fprim"]
        results["tprim"] = nml["species_parameters_1"]["tprim"]
        
        return results

def params():
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
    return {
        "species_parameters_1::fprim":   cp.Uniform(0.8 * 2.2, 1.2 * 2.2),  # +/- 20%
        "species_parameters_1::tprim":    cp.Uniform(0.8 * 6.9, 1.2 * 6.9)
    }

campaign = Campaign(name="load_results_pce_order_6")
input_decoder = GS2InputDecoder(target_filename="flsm_new.in")
output_decoder = GS2OutputDecoder(target_filename="flsm_new.out.nc")

pce_order = 6
nsamples = (1 + pce_order) ** 2
input_files = [f"/home/userfs/b/bc1264/Documents/pce_test_again/runs/runs_0-100000000/runs_0-1000000/runs_0-10000/runs_0-100/run_{i}/flsm_new.in" for i in range(1, nsamples+1)]
output_files = [f"/home/userfs/b/bc1264/Documents/pce_test_again/runs/runs_0-100000000/runs_0-1000000/runs_0-10000/runs_0-100/run_{i}/flsm_new.out.nc" for i in range(1, nsamples+1)]

campaign.add_app(name="readpls", params=params())
campaign.set_sampler(PCESampler(vary=vary(), polynomial_order=6))
campaign.add_external_runs(input_files=input_files, output_files=output_files, input_decoder=input_decoder, output_decoder=output_decoder)

results_df = campaign.get_collation_result()
results = campaign.analyse(qoi_cols=["ky", "omega/4", "gamma"])

# Extract and pickle results using Dimits normalisation (not GS2's)
ky = results.describe("ky", "mean") / np.sqrt(2)
omega = results.describe("omega/4", "mean") * (-np.sqrt(2) / 2.2)
omega_std = results.describe("omega/4", "std") * (np.sqrt(2) / 2.2)
gamma = results.describe("gamma", "mean") * (np.sqrt(2) / 2.2)
gamma_std = results.describe("gamma", "std") * (np.sqrt(2) / 2.2)
sobols_first_omega = results.sobols_first()["omega/4"]
sobols_first_gamma = results.sobols_first()["gamma"]
sobols_total_omega = results.sobols_total()["omega/4"]
sobols_total_gamma = results.sobols_total()["gamma"]

processed_results = {"ky": ky,
                        "omega": omega,
                        "omega_std": omega_std,
                        "gamma": gamma,
                        "gamma_std": gamma_std,
                        "sobols_first_omega": sobols_first_omega, 
                        "sobols_first_gamma": sobols_first_gamma,
                        "sobols_total_omega": sobols_total_omega,
                        "sobols_total_gamma": sobols_total_gamma}

with open(f"processed_results_pce_order_{pce_order}.pickle", "wb") as f:
    pickle.dump(processed_results, f)