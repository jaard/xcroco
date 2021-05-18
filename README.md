# xcroco
Set of python tools to analyse output from the CROCO/ROMS ocean model using [xarray](https://github.com/pydata/xarray).
Heavily inspired by the original ROMSTOOLS Matlab code (Penven et al., 2007, https://doi.org/10.1016/j.envsoft.2007.07.004), with function names being largely compatible. Some inspiration was also taken from the [xroms](https://github.com/bjornaa/xroms) package.

Basic documentation is in preparation, please be patient and rely on the docstrings of individual methods in the meantime.

## How to install
I strongly recommend to use Miniconda or Anaconda environments to manage your Python packages. However, xcroco is not available through these channels yet. To avoid problems with installing xcroco dependencies manually, make sure to install the following packages in your Conda environment before installing xcroco:

```
conda install xarray numpy scipy cartopy
conda install -c conda-forge xgcm
```

Next, clone this repository to a directory of your choice (e.g. `/Users/your-username/your-code-folder/`) with 

```
cd /Users/your-username/your-code-folder/
git clone https://github.com/jaard/xcroco.git
```

If you dont have [ssh keys](https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account) set up, you may be asked to enter your GitHub password.
After cloning the repository, check that `which pip` points to your Conda environment. If this is not already the case, activate the desired environment with `source ~/miniconda3/bin/activate your-conda-environment`. Now you can install xcroco from source using

```
pip install -e xcroco
```

## Notebook with usage examples

To see some examples of some basic analysis of  CROCO output using xcroco, please see the files [xcroco_examples.ipynb](https://raw.githubusercontent.com/jaard/xcroco/master/xcroco_example.ipynb) and [xcroco_examples.html](https://raw.githubusercontent.com/jaard/xcroco/master/xcroco_example.html) which have identical content. The ipynb version can be edited an run using Jupyter Lab, while you can view the .html version with any browser. The github website cannot show large HTML files directly, so please download and view it locally. 


