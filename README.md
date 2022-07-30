## MSc Project

The working title of my project is _Uncertainty quantification of ion temperature gradient modes_. The aim is to calculate the uncertainty in the frequency $\omega_r$ and growth rate $\gamma$ of ion temperature gradient modes calculated by the gyrokinetic code [GS2](https://gyrokinetics.gitlab.io/gs2/). I'll use the Python library [EasyVVUQ](https://easyvvuq.readthedocs.io/en/dev/) to fascilitate this. 

### Running this script
First clone this repository:
```
git clone https://github.com/baiway/MSc-project.git
```
Run tests (these need to be improved):
```
python test_gs2uq.py
```
Review arguments passed to `run_campaign()`. You'll need to change `gs2_bin` to your GS2 path. You may also wish to change `nprocs` and `pce_order` (only polynomial chaos sampling is supported at the moment). Once these are set, run the script with
```
python run_uq.py
```
The final plot(s) will be displayed once all the runs are complete. They will also be saved in the root folder of this project.
