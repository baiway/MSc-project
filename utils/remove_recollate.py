""" Currently non-functional. You cannot just remove runs without replacing them
EasyVVUQ needs the correct number of runs to analyse the data.
"""

from easyvvuq import Campaign
import matplotlib.pyplot as plt
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
        results = {"fprim": None, "tprim": None, "pk": None, "shat": None}

        run_dir = Path(run_info['run_dir'])
        input_filepath = run_dir / self.target_filename

        print(input_filepath.resolve())

        # Open template filename
        with open(input_filepath, "r") as f:
            nml = f90nml.read(f)

        results["fprim"] = nml["species_parameters_1"]["fprim"]
        results["tprim"] = nml["species_parameters_1"]["tprim"]
        results["pk"] = nml["theta_grid_parameters"]["pk"]
        results["shat"] = nml["theta_grid_parameters"]["shat"]

        return results

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
            "default": 6.9},
        "theta_grid_parameters::pk": {
            "type": "float", 
            "min": 0.8*1.44, 
            "max": 1.2*1.44, 
            "default": 1.44},
        "theta_grid_parameters::shat": {
            "type": "float", 
            "min": 0.8*0.8, 
            "max": 1.2*0.8, 
            "default": 0.8}
    }

def vary():
    """Define distributions to sample over for each varying quantity in format
        vary = {
            "group::param": cp.Distribution
        }
    using ChaosPy distributions. See: https://chaospy.readthedocs.io/en/master/user_guide/index.html 
    """

    return {
        "species_parameters_1::fprim":   cp.Uniform(0.8 * 2.2, 1.2 * 2.2),  # +/- 20%
        "species_parameters_1::tprim":    cp.Uniform(0.8 * 6.9, 1.2 * 6.9),
        "theta_grid_parameters::pk":   cp.Uniform(0.8 * 1.44, 1.2 * 1.44),
        "theta_grid_parameters::shat":    cp.Uniform(0.8 * 0.8, 1.2 * 0.8)
    }

def plot_all_sobols(filename):
    q = filename.split("processed_results_")[1].split(".pickle")[0]
    plotname = q+"_sobols.png"

    R = {}

    try:
        with open(filename, "rb") as f:
            R["results"] = pickle.load(f)
    
    except FileNotFoundError:
        print("Could not find file: ", filename)

    # Extract results
    ky = R["results"]["ky"]
    sobols_first_omega = R["results"]["sobols_first_omega"]
    sobols_first_gamma = R["results"]["sobols_first_gamma"]
    sobols_second_omega = R["results"]["sobols_second_omega"]
    sobols_second_gamma = R["results"]["sobols_second_gamma"]
    sobols_total_omega = R["results"]["sobols_total_omega"]
    sobols_total_gamma = R["results"]["sobols_total_gamma"]

    # LaTeX for plot legends
    symbols = {
        "fprim": r"$\kappa_n$",
        "tprim": r"$\kappa_t$",
        "pk": r"$p_k$",
        "shat": r"$\hat{s}$"
    }

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8, 8.5))
    #fig.suptitle("First order Sobol indices")

    for k in sobols_first_omega.keys():
        param = k.split("::")[1] # GS2 label
        symbol = symbols[param]
        axs[0, 0].plot(ky, sobols_first_omega[k], "o-", label=symbol)
        axs[0, 1].plot(ky, sobols_first_gamma[k], "o-", label=symbol)

    plotted_pairs = []

    for k1 in sobols_second_omega.keys():
        for k2 in sobols_second_omega[k1].keys():
            p1, p2 = k1.split("::")[1], k2.split("::")[1]
            if ((p1, p2) in plotted_pairs) or ((p2, p1) in plotted_pairs): 
                continue
            else:
                s1, s2 = symbols[p1], symbols[p2]
                plotted_pairs.append((p1, p2))
                axs[1, 0].plot(ky, sobols_second_omega[k1][k2], "o-", label=s1+", "+s2)
                axs[1, 1].plot(ky, sobols_second_gamma[k1][k2], "o-", label=s1+", "+s2)

    for k in sobols_total_omega.keys():
        param = k.split("::")[1] # GS2 label
        symbol = symbols[param]
        axs[2, 0].plot(ky, sobols_total_omega[k], "o-", label=symbol)
        axs[2, 1].plot(ky, sobols_total_gamma[k], "o-", label=symbol)

    axs[0, 0].set_ylabel("First order Sobol indices")
    axs[1, 0].set_ylabel("Second order Sobol indices")
    axs[2, 0].set_ylabel("Total Sobol indices")
    axs[0, 0].set_title(r"$\omega_r/4$")
    axs[0, 1].set_title(r"$\gamma$")
    fig.supxlabel(r"$k_y\rho$")

    axs[0, 0].annotate("(a)", xy=(0.20, 0.85))
    axs[0, 1].annotate("(b)", xy=(0.05, 0.85))
    axs[1, 0].annotate("(c)", xy=(0.05, 0.85))
    axs[1, 1].annotate("(d)", xy=(0.05, 0.85))
    axs[2, 0].annotate("(e)", xy=(0.20, 0.85))
    axs[2, 1].annotate("(f)", xy=(0.05, 0.85))

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[2, 0].legend()
    axs[2, 1].legend()

    plt.savefig(plotname, dpi=300)
    plt.show()

def plot_means(filename):
    q = filename.split("processed_results_")[1].split(".pickle")[0]
    plotname = q+"_means.png"

    R = {}

    try:
        with open(filename, "rb") as f:
            R["results"] = pickle.load(f)
    
    except FileNotFoundError:
        print("Could not find file: ", filename)

    # Extract results
    ky = R["results"]["ky"]
    omega = R["results"]["omega"]
    omega_std = R["results"]["omega_std"]
    gamma = R["results"]["gamma"]
    gamma_std = R["results"]["gamma_std"]

    plt.figure()
    plt.plot(ky, omega, "o-", color="orange", label=r"$\omega_r/4$")
    plt.plot(ky, omega - omega_std, "--", color="orange")
    plt.plot(ky, omega + omega_std, "--", color="orange")
    plt.fill_between(ky, omega - omega_std, omega + omega_std, color="orange", alpha=0.4)
    plt.plot(ky, gamma, "o-", color="blue", label=r"$\gamma$")
    plt.plot(ky, gamma - gamma_std, "--", color="blue")
    plt.plot(ky, gamma + gamma_std, "--", color="blue")
    plt.fill_between(ky, gamma - gamma_std, gamma + gamma_std, color="blue", alpha=0.2)
    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel(r"$\gamma$" + " " + r"$[v_\mathrm{th}/L_{ne}]$")
    plt.savefig(plotname, dpi=300)
    plt.close()

if __name__ == "__main__":
    campaign = Campaign(name="recollate_attempt")
    input_decoder = GS2InputDecoder(target_filename="flsm_new.in")
    output_decoder = GS2OutputDecoder(target_filename="flsm_new.out.nc")

    pce_order = 3
    nsamples = (1 + pce_order) ** 2

    input_files, output_files = [], []
    base_dir = "/Users/baileycook/Downloads/fprim_tprim_pk_shat/MSc-project/itg_uq_h1b3ca8/runs/runs_0-100000000/runs_0-1000000/runs_0-10000/"
    excluded_runs = {
        "runs_0-100/": [4,8,15,28,32,68,72,76,80,92,96],
        "runs_100-200/": [131,136,140,144,160],
        "runs_200-300/": [200,208]
    }

    for id in range(1, 256+1):
        if id < 100: 
            folder = "runs_0-100/"
        elif 100 <= id < 200: 
            folder = "runs_100-200/"
        elif id >= 200: 
            folder = "runs_200-300/"

        if id in excluded_runs[folder]:
            print(f"Skipping run {id}")
            continue
        else:
            inpath = Path(base_dir + folder + f"run_{id}/flsm_new.in").resolve()
            outpath = Path(base_dir + folder + f"run_{id}/flsm_new.out.nc").resolve()
            input_files.append(inpath)
            output_files.append(outpath)
                
    campaign.add_app(name="readpls", params=params())
    campaign.set_sampler(PCESampler(vary=vary(), polynomial_order=2))
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
    sobols_second_omega = results.sobols_first()["omega/4"]
    sobols_second_gamma = results.sobols_first()["gamma"]
    sobols_total_omega = results.sobols_total()["omega/4"]
    sobols_total_gamma = results.sobols_total()["gamma"]

    processed_results = {"ky": ky,
                        "omega": omega,
                        "omega_std": omega_std,
                        "gamma": gamma,
                        "gamma_std": gamma_std,
                        "sobols_first_omega": sobols_first_omega, 
                        "sobols_first_gamma": sobols_first_gamma,
                        "sobols_second_omega": sobols_second_omega, 
                        "sobols_second_gamma": sobols_second_gamma,
                        "sobols_total_omega": sobols_total_omega,
                        "sobols_total_gamma": sobols_total_gamma}

    with open(f"cleaned_up_results.pickle", "wb") as f:
        pickle.dump(processed_results, f)

    plot_means("cleaned_up_results.pickle")
    plot_all_sobols("cleaned_up_results.pickle")