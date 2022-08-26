## MSc Project

The title of my project is _Uncertainty quantification of ion temperature gradient modes_. The aim is to determine the uncertainty in the frequency $\omega_r$ and growth rate $\gamma$ of ion temperature gradient modes (ITGs) calculated by the gyrokinetic code [GS2](https://gyrokinetics.gitlab.io/gs2/). I use the Python library [EasyVVUQ](https://easyvvuq.readthedocs.io/en/dev/) to fascilitate this. 

The code is reasonably well-commented throughout; separate documentation does not exist though. 

### Running the code
First you'll need to install [EasyVVUQ](https://easyvvuq.readthedocs.io/en/dev/), [GS2](https://gyrokinetics.gitlab.io/gs2/) and their dependencies. Installation instructions can be found on the linked webpages. 
Next, clone this repository:
```
git clone https://github.com/baiway/MSc-project.git
```
Run tests:
```
python test_gs2uq.py
```
Ensure you check the arguments passed to `run_campaign()`. You'll need to change `gs2_bin` to your GS2 path. You may also wish to change `nprocs` and `pce_order` (only polynomial chaos sampling is supported). Once these are set, run the script with
```
python run_uq.py
```
The final plots (examples below) will be displayed once all the runs are complete. They will also be saved in the root folder of this project.

![Mean frequencies $\omega_r/4$ and growth rates $\gamma$ plotted as a function of wavenumber $k_y\rho$. The error bars show standard deviations.](https://github.com/baiway/MSc-project/blob/main/first-actual-scan.png?raw=true)

![First order Sobol index of the frequency $\omega_r/4$ plotted as a function of wavenumber $k_y\rho$.](https://github.com/baiway/MSc-project/blob/main/first_Sobol_omega_attempt1.png?raw=true)

![First order Sobol index of the growth rate $\gamma$ plotted as a function of wavenumber $k_y\rho$.](https://github.com/baiway/MSc-project/blob/main/first_Sobol_gamma_attempt1.png?raw=true)
