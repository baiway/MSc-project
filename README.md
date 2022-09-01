## MSc Project

The title of my project is _Uncertainty quantification of ion temperature gradient modes_. The aim is to determine the uncertainty in the frequency $\omega_r$ and growth rate $\gamma$ of ion temperature gradient modes (ITGs) calculated by the gyrokinetic code [GS2](https://gyrokinetics.gitlab.io/gs2/). I use the Python library [EasyVVUQ](https://easyvvuq.readthedocs.io/en/dev/) to fascilitate this. 

The code is reasonably well-commented throughout; separate documentation does not exist though. 

### Running the code
First you'll need to install [EasyVVUQ](https://easyvvuq.readthedocs.io/en/dev/), [GS2](https://gyrokinetics.gitlab.io/gs2/) and their dependencies. Installation instructions can be found on the linked webpages. 
Next, clone this repository:
```
git clone https://github.com/baiway/MSc-project.git
```
Run tests (note: these only check that `GS2Encoder` and `GS2Decoder` in `gs2uq.py` work as intended):
```
python test.py
```
Ensure you check the arguments passed to `run_campaign()`. You'll need to change `gs2_bin` to the full path to your GS2 bin. You may also wish to change `nprocs` and `pce_order` to suitable values (I do not recommend going above `pce_order=5`). Once these are set, run the script with
```
python run_uq.py
```
The final plots will be displayed once all the runs are complete. They will also be saved in the root folder of this project. Explains of plots produced using `pretty_plots.py` in the `utils` folder are shown below.

![Mean values of the real frequency $\omega_r/4$ and growth rate $\gamma$ for mode numbers $k_y\rho=0.05$ to $k_y\rho=0.50$. The error bars are the standard deviations in $\omega_r/4$ and $\gamma$ arising from varying the density gradient $\kappa_n$, temperature gradient $\kappa_T$, magnetic shear $\hat{s}$, and safety factor $q$ (through $p_k=2L_\mathrm{ref}/qR$) by $\pm 20\%$. The varying parameter space is sampled with PCE using polynomial order 3. A total of 256 samples were made; the full simulation took around 20 hours to complete on 16 processors.](https://github.com/baiway/MSc-project/blob/main/example_plots/fprim_tprim_pk_shat_means.png?raw=true)

![First, second and total Sobol indices associated with varying $\kappa_n$, $\kappa_T$, $\hat{s}$ and $p_k=2L_\mathrm{ref}/qR$ by $\pm 20\%$.](https://github.com/baiway/MSc-project/blob/main/example_plots/fprim_tprim_pk_shat_sobols.png?raw=true)

<figure>
  <img
  src="https://github.com/baiway/MSc-project/blob/main/example_plots/fprim_tprim_pk_shat_means.png?raw=true"
  alt="Dimits et al. plot with error bars.">
  <figcaption>
    Mean values of the real frequency $\omega_r/4$ and growth rate $\gamma$ for mode numbers $k_y\rho=0.05$ to $k_y\rho=0.50$. The error bars are the standard deviations in $\omega_r/4$ and $\gamma$ arising from varying the density gradient $\kappa_n$, temperature gradient $\kappa_T$, magnetic shear $\hat{s}$, and safety factor $q$ (through $p_k=2L_\mathrm{ref}/qR$) by $\pm 20\%$. The varying parameter space is sampled with PCE using polynomial order 3. A total of 256 samples were made; the full simulation took around 20 hours to complete on 16 processors.
  </figcaption>
</figure>
