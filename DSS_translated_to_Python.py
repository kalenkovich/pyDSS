
# coding: utf-8

# This is a test of the translation of the DSS algorithm from Matlab to Python. The Matlab source is taken from `NoiseTools` package that can be found [here](http://audition.ens.fr/adc/NoiseTools/). 
# 
# The test consists of running an example from the `NoiseTools` in parallel in Matlab and Python and matching the results.

# # Setup

# ## Imports

from pathlib import Path
import inspect


# Code highlightintg
from pygments import highlight
from pygments.lexers import MatlabLexer, PythonLexer
from pygments.formatters import HtmlFormatter


from IPython.core.display import HTML


# This is a package that allows us to inerface with a Matlab process
from pymatbridge import Matlab


import numpy as np
import scipy
from matplotlib import pyplot as plt


# ## Set up autoreloading of the Python code

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1  # autoreload only the source loaded with %aimport')
get_ipython().run_line_magic('aimport', 'dss')


# ## Paths

# Change this to wherever you unpacked `NoiseTools` to.

noise_tools_dir = Path('NoiseTools')


noise_tools_examples_dir = noise_tools_dir / 'EXAMPLES' / 'a few old example scripts'
example_1 = noise_tools_examples_dir / 'example1.m'


# ## Printing code

def print_highlighted_code(code, lexer):
    display(HTML(highlight(code, lexer, HtmlFormatter(full=True, linenos='inline'))))
    
def print_matlab_code(code):
    print_highlighted_code(code, MatlabLexer())
    
def print_python_code(code):
    print_highlighted_code(code, PythonLexer())

def print_matlab_script(path):
    with open(path, 'r') as f:
        code = f.read()
    print_matlab_code(code)
    
def print_python_function(fun):
    code = inspect.getsource(fun)
    print_python_code(code)


# ## Start Matlab

matlab = Matlab()
matlab.start()


# ## Add `NoiseTools` to the Matlab's path

matlab.run_code('addpath(\'{}\')'.format(noise_tools_dir))


# # Simulate data

# Let's look at the example 1 code:

print_matlab_script(example_1)


# Let's create synthetic data in Matlab and transfer it here.

example_1_code = open(example_1, 'r').readlines()
synthethize_data_code = ''.join(example_1_code[9:21])
print_matlab_code(synthethize_data_code)


matlab.run_code(synthethize_data_code)
data = matlab.get_variable('data')
print(data.shape)


# That is 300 time points, 30 channels and 100 trials.

# # Calculate covariances

# ## Inspect the `NoiseTools` code

# Here is how the covariance matrices are calculated:

covariances_code = ''.join(example_1_code[22:24])
print_matlab_code(covariances_code)


# Let's see what `nt_cov` does.

nt_cov = noise_tools_dir / 'nt_cov.m'
print_matlab_script(nt_cov)


# The example code only uses the first argument `x` and only the first output `c`. Also, both `data` and `mean(data,3)` are single matrices, not arrays of matrices. Therefore, checks on line 33 (`isempty(w)`) and on line 35 (`is_numberic(x)`) both avaluate to `true`. So, the only piece of code relevant to us is this:

nt_cov_code = open(nt_cov, 'r').readlines()
print_matlab_code(''.join(nt_cov_code[35:42]))


# Let's see what `nt_multishift` does. First, we need to figure out what `shifts` is since we didn't supply this parameter. Here are the relevant lines:

print_matlab_code(''.join(nt_cov_code[26:29]))


# So, shifts is just `0`. Let's see what `nt_multishift` does in this case.

nt_multishift_path = noise_tools_dir / 'nt_multishift.m'
print_matlab_script(nt_multishift_path)


# And here is the relevant part:

nt_multishift_code = open(nt_multishift_path, 'r').readlines()
print_matlab_code(''.join(nt_multishift_code[28:32]))


# So, with `shifts == 0` function `nt_multishift` does not change the input. Let's get back to the `nt_cov` code then.

nt_cov_code = open(nt_cov, 'r').readlines()
print_matlab_code(''.join(nt_cov_code[35:42]))


# Since `data` is shaped as `nsamples`-by-`nchans`-by-`ntrials` then so is `x`.
# 
# For each `k`:
# - `xx` is the data from one trial,
# - `xx'*xx` is uncentered covariance of channels in time during a single trial.
# 
# Then `c` is the sum of per-trial ucnentered covariances.

# ## Run the Matlab code

print_matlab_code(covariances_code)


matlab.run_code(covariances_code)
c0 = matlab.get_variable('c0')
c1 = matlab.get_variable('c1')


# ## Run the Python code

print_python_function(dss.calculate_covariances)


R0, R1 = dss.calculate_covariances(data, uncentered=True, unnormalized=True)


# ## Match

np.allclose(R0, c0)


np.allclose(R1, c1)


# # PCA

# The DSS algorithm includes two PCA rotations. Let's check that we do it in the same manner.

# ## Take a look at the `NoiseTools` code

pca_path = noise_tools_dir / 'nt_pcarot.m'
print_matlab_script(pca_path)


# We won't be working with sparse matrices so the Python code won't have the `N` parameter.

# ## Take a look at the Python code

print_python_function(dss.covariance_pca)


# ## Run the code

# ### Matlab

matlab.run_code('[E, D] = nt_pcarot(c0);')
E = matlab.get_variable('E')
D = matlab.get_variable('D')


# ### Python

eigvals, eigvecs = dss.covariance_pca(R0)


# ## Match

np.allclose(eigvals, D)


np.allclose(eigvecs, E)


# Let's take a look at the topleft corners.

eigvecs[:5, :5]


E[:5, :5]


# The first problem is some vectors differ in sign. Let's coerce the first row of both matrices to be positive.

eigvecs *= np.sign(eigvecs[np.newaxis, 0])
E = E * np.sign(E[np.newaxis, 0])


np.allclose(eigvecs, E)


# The second problem is that eigenvectors corresponding to the very small eigenvalues may differ a lot.

threshold = 1e-9
abs_threshold = max(eigvals) * threshold
np.allclose(eigvecs[:, eigvals > abs_threshold], E[:, D.squeeze() > abs_threshold])


# So, the results are equivalent. Let's check that both functions produce the same output if we ask them to remove eigenvalues below a given threshold.

matlab.set_variable(value=threshold, varname='threshold')
matlab.run_code('[E, D] = nt_pcarot(c0, [], threshold);')
E = matlab.get_variable('E')
D = matlab.get_variable('D')
eigvals, eigvecs = dss.covariance_pca(R0, threshold=threshold)


np.allclose(eigvals, D)


eigvecs *= np.sign(eigvecs[np.newaxis, 0])
E = E * np.sign(E[np.newaxis, 0])
np.allclose(eigvecs, E)


# # DSS from covariance matrices

# Here is how the dss is run in the example:

dss_code = ''.join(example_1_code[24:26])
print_matlab_code(dss_code)


# Two steps:
# 1. `nt_dss0` calculates the unmixing matrix,
# 2. `nt_mmat` applies it to the data.
# 
# We'll add calculating the mixing matrix to the first step as well - it will give us the topographies of the components.

# ## Unmixing and mixing

# ### Matlab code

# Let's see what `nt_dss0` does.

nt_dss0_path = noise_tools_dir / 'nt_dss0.m'
print_matlab_script(nt_dss0_path)


# ### Python code

print_python_function(dss.unmix_covariances)


# ### Run code

# #### Matlab

unmxing_matlab_code = dss_code.split('\n')[0]
print_matlab_code(unmxing_matlab_code)


matlab.run_code(unmxing_matlab_code)
todss = matlab.get_variable('todss')
pwr0 = matlab.get_variable('pwr0')
pwr1 = matlab.get_variable('pwr1')


# #### Python

c0 = matlab.get_variable('c0')
c1 = matlab.get_variable('c1')
U, M, phase_locked_power, total_power = dss.unmix_covariances(c0, c1, threshold=1e-9, return_mixing=True, return_power=True)
# 1e-9 is the default threshold used in the Matlab code, see line 16


# ### Match

# Filters only need to match up to a sign of columns.

np.allclose(
    U * np.sign(U[np.newaxis, 0]),
    todss * np.sign(todss[np.newaxis, 0]),
)


# And they do.

# Matlab code takes a square root after power calculation so we'll do that as well before matching.

np.allclose(pwr0, np.sqrt(phase_locked_power))


np.allclose(pwr1, np.sqrt(total_power))


# ### Check the unmixing matrix

dss.is_pseudoinverse(U, M)

