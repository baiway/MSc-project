"""To run interactively, type `ipython --pylab='qt'`
 then `import xarray as xr`.

 Run GS2 with nice -n 10 mpirun -n 6 /home/userfs/b/bc1264/Documents/gs2/bin/gs2 flsm_comb.in
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load Dimits data
dd1 = np.loadtxt("Dimits_omega.txt", delimiter=",")
dky1, Dimits_omega = dd1[:,0], dd1[:,1]
dd2 = np.loadtxt("Dimits_gamma.txt", delimiter=",")
dky2, Dimits_gamma = dd2[:,0], dd2[:,1]

# Load GS2 data
with xr.open_dataset("flsm_comb.out.nc", engine="netcdf4") as ds:
    GS2_ky = ds.ky.data / np.sqrt(2)
    GS2_omega = (ds.omega_average.isel(ri=0, t=-1).squeeze().data * (-np.sqrt(2) / 2.2)) / 4
    GS2_gamma = ds.omega_average.isel(ri=1, t=-1).squeeze().data * (np.sqrt(2) / 2.2)

# Plot Dimits data
plt.plot(dky1, Dimits_omega, "o--", label="Dimits", color="k")
plt.plot(dky2, Dimits_gamma, "o--", color="r")

# Plot GS2 data
plt.plot(GS2_ky, GS2_omega, "o-", label="GS2", color="k")
plt.plot(GS2_ky, GS2_gamma, "o-", color="r")

# Label axes, add text to plot and add legend
plt.legend()
plt.xlabel(r"$k_y\rho$")
plt.ylabel(r"$\gamma$" + " " + r"$[v_{thr}/a]$")
plt.text(0.53, 0.06, r"$\gamma$", color="r", fontsize=14)
plt.text(0.53, -0.125, r"$\omega/4$", color="k", fontsize=14)
plt.legend(loc="lower left")

# Save and display plot
plt.savefig(f"freq_and_growth_rate_final.png", dpi=300)
plt.show()