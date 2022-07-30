#! /usr/bin/env python

from typing import Any
from pathlib import Path
import datetime
import shutil
import os
import subprocess
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from gs2uq import GS2Encoder

class GS2Scan:
    """
    Scanning script to check for convergence
    """
    def __init__(self, vary: dict) -> None:
        self.vary_dict = vary
        self.param_names = [p.split("::")[1] for p in vary.keys()]
    
    def write_inputs(self) -> None:
        for param in self.param_names:
            target_dir = Path(__file__).parent / param
            if target_dir.is_dir():
                if self._should_overwrite(target_dir):
                    self._overwrite(target_dir)
                else:
                    continue
            else:
                target_dir.mkdir()
                self._populate(target_dir)

    def run_GS2(self, nproc=8, gs2_bin="/home/userfs/b/bc1264/Documents/gs2/bin/gs2") -> None:
        if not Path(gs2_bin).is_file():
            raise FileExistsError(f"GS2 not found in: {gs2_bin}")
        
        for param in self.param_names:
            param_dir = Path(__file__).parent / param
            files = os.listdir(param_dir)
            input_files = [file for file in files if ".in" in file]

            for file in input_files:
                file_path = param_dir / file
                stdout = open(param_dir / f"{file}".replace("input", "print"), "w")
                subprocess.run(f"nice -n 10 mpirun -n {nproc} {gs2_bin} {file_path}", 
                               stdout=stdout, shell=True)

    def get_output(self, scan_param: str = None, plot_phi2=True, return_results=True) -> dict:
        if scan_param is None:
            print("You need to provide a scan parameter")

        scan_dir = Path(__file__).parent / scan_param
        files = os.listdir(scan_dir)
        output_files = [file for file in files if ".out.nc" in file]
        
        results = {file: None for file in output_files}

        for file in output_files:
            file_path = scan_dir / file

            result = {"ky": None, "omega/4": None, "gamma": None}

            with xr.open_dataset(file_path, engine="netcdf4") as ds:
                if plot_phi2: 
                    t = ds.t.data
                    phi2 = ds.phi2.squeeze().data
                    self._plot_phi2(t, phi2, file)
                
                if return_results:
                    result["ky"] = ds.ky.data
                    result["omega/4"] = ds.omega_average.isel(ri=0, t=-1).squeeze().data / 4
                    result["gamma"] = ds.omega_average.isel(ri=1, t=-1).squeeze().data
                    
                    results["file"] = result
        
                    return results

    def _get_full_key(self, param: str) -> str:
        for full_key in self.vary_dict.keys():
            if param in full_key:
                return full_key
            else:
                continue

    def _should_overwrite(self, target_dir) -> bool:
        print(f"{target_dir} already exists.")
        while True:
            proceed = input("Would you like to overwrite this directory? (y/n) \n > ")
            if proceed.lower() == "y": 
                return True

            elif proceed.lower() == "n":
                return False

            else:
                print("Invalid input.")
    
    def _overwrite(self, target_dir) -> None:
        print(f"Overwriting directory: {target_dir}")
        shutil.rmtree(target_dir)
        target_dir.mkdir()
        self._populate(target_dir)

    def _populate(self, target_dir) -> None:
        param = target_dir.name
        key = self._get_full_key(param)
        value = self.vary_dict[key]     # either a value or array of values

        if self._is_iter(value):
            for v in value:
                id = f"{param}_{v}"
                encoder = GS2Encoder(template_filename="./flsm_comb.in",
                                     target_filename=f"GS2_input_{id}.in")
                encoder.encode({key: v}, target_dir=target_dir)
        else:
            id = f"{param}_{value}"
            encoder = GS2Encoder(template_filename="./flsm_comb.in",
                                 target_filename=f"GS2_input_{id}.in")
            encoder.encode({key: value}, target_dir=target_dir)
        
    @staticmethod
    def _is_iter(obj: Any) -> bool:
        if isinstance(obj, (list, np.ndarray)):
            return True
        elif isinstance(obj, (int, np.integer, float, np.floating)):
            return False
        else:
            raise ValueError(f"Unsupported type: {type(obj)}")
    
    @staticmethod
    def _plot_phi2(t, phi2, filename) -> None:
        plt.figure()
        plt.plot(t, phi2)
        plt.xlabel("t [a/v_thr]")
        plt.ylabel(r"$\phi^2$" + " " + r"$[(T_r/e)^2]$")
        plt.yscale("log")
        plt.title(filename)
        plt.savefig(filename, dpi=300)
        plt.clf()

if __name__ == "__main__":
    vary = {
        #"theta_grid_parameters::ntheta": np.arange(1, 10), 
        "theta_grid_parameters::nperiod": np.arange(2, 8),
        #"le_grids_knobs::ngauss": 5,
        #"le_grids_knobs::negrid": 30
    }

    myscan = GS2Scan(vary)
    myscan.write_inputs()
    myscan.run_GS2()
    qoi = "ntheta"
    time = datetime.datetime.now()
    ntheta_results = myscan.get_output(qoi, plot_phi2=True, return_results=True)
    
    plt.figure()
    for filename, result in ntheta_results.items():
        ky = result["ky"] / np.sqrt(2)
        omega = result["omega/4"] * (-np.sqrt(2) / 2.2)
        gamma = result["gamma"] * (np.sqrt(2) / 2.2)
        plt.plot(ky, omega, "o-", label=r"$\omega}_r/4$" + f" from {filename}")
        plt.plot(ky, gamma, "o-", label=r"$\gamma$" + f" from {filename}")
    
    plt.legend()
    plt.xlabel(r"$k_y\rho$")
    plt.ylabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")
    plt.savefig(f"freq_and_growth_rate_{qoi}_{time}.png", dpi=300)
    plt.show()
    plt.clf()