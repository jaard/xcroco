# xcroco
Set of python tools to analyse output from the CROCO/ROMS ocean model using [xarray](https://github.com/pydata/xarray).
Heavily inspired by the original ROMSTOOLS Matlab code (Penven et al., 2007, https://doi.org/10.1016/j.envsoft.2007.07.004), with function names being largely compatible. Some inspiration regarfing horizontal interpolation was taken from the [xroms](https://github.com/bjornaa/xroms) package.

Documentation is currently missing, please rely on the docstrings of individual methods for hints on usage.

## How to install
Clone this repository with `git clone git@github.com:jaard/xcroco.git` and
install it from source using `python setup.py install` or `pip install .`

If you use the Anaconda package manager, make sure you use the correct environment by checking that `which python` or `which pip` point to your miniconda directory.
If that is not already the case, first activate the desired environment with `source ~/miniconda3/bin/activate your-conda-environment`.

>If you dont have [ssh keys](https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account) set up, you can use `$ git clone https://github.com/jaard/xcroco.git` and enter your github password.

