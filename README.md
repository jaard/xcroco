# xcroco
Set of python tools to analyse output from the CROCO/ROMS ocean model using [xarray](https://github.com/pydata/xarray).
Heavily inspired by the original ROMSTOOLS Matlab code (Penven et al., 2007, https://doi.org/10.1016/j.envsoft.2007.07.004), with function names being largely compatible. Some inspiration was also taken from the [xroms](https://github.com/bjornaa/xroms) package.

Basic documentation is in preparation, please be patient and rely on the docstrings of individual methods in the meantime.

## How to install
I strongly recommended to use Miniconda or Anaconda environments to manage your Python packages, however xcroco is not available through these channels yet. To avoid problems with installing xcroco dependencies manually, make sure to install the following packages in your Conda environment before installing xcroco:

```
conda install xarray numpy scipy
conda install -c conda-forge xgcm
```

Next, clone this repository with 

```
git clone git@github.com:jaard/xcroco.git
```

If you dont have [ssh keys](https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account) set up, you can use `$ git clone https://github.com/jaard/xcroco.git` and enter your github password.
Check that `which python` or `which pip` point to your Conda environment directory. If that is not already the case, first activate the desired environment with `source ~/miniconda3/bin/activate your-conda-environment`.
Now you can install xcroco from source using

```
pip install -e .
```
