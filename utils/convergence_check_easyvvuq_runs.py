"""Plot phi2 against t for EasyVVUQ samples to check for convergence
"""


import xarray as xr
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

def get_plots(filepath=None, run_num=None, plot_phi2=True):
    R = {}
    try:
        with xr.open_dataset(filepath, engine="netcdf4") as ds:
            phi2 = ds.phi2.squeeze().data
            t = ds.t.data
            #omega = ds.omega_average.isel(ri=0).squeeze().data / 4
            #gamma = ds.omega_average.isel(ri=1).squeeze().data
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {filepath}")
    
    if plot_phi2:
        plt.semilogy(t, phi2)
        plt.xlabel(r"$t [a/v_thr]$")
        plt.ylabel(r"$\phi^2$" + " " + r"$[(T_r/e)^2]$")
        plt.title(f"Run {run_num}")
        plt.show()
    
    # do stuff
    pass

if __name__ == "__main__":
    run_dir = "~/Downloads/fprim_tprim_pk_shat/MSc-project/itg_uq_h1b3ca8/runs/runs_0-100000000/runs_0-1000000/runs_0-10000/"
    outfile_name = "flsm_new.out.nc"
    num_runs = 256

    for i in range(200, num_runs+1):
        if i < 100: 
            join_dir = "runs_0-100/run_"
        elif 100 <= i < 200: 
            join_dir = "runs_100-200/run_"
        elif i >= 200: 
            join_dir = "runs_200-300/run_"

        filepath = run_dir + join_dir + str(i) + "/" + outfile_name
        get_plots(filepath, i, plot_phi2=True)
