# %%
# ## Example using constrained GP model
# This is the code used to produce the first example in the paper _'Gaussian processes with linear operator inequality constraints'_, https://arxiv.org/abs/1901.03134

### Basic imports ###
import sys, os

# For plotting
import plotly
import plotly.offline as pltlyoff
from IPython.display import display, HTML

# This is for plotting as static images (to show on e.g. GitHub)
import plotly.io as pio
from IPython.display import Image

# Numerics
import numpy as np
import scipy as sp
import itertools
import pyDOE

### Custom files ###

# Path to custom plotly module 'GPPlotly' for plotting
# can be downloaded at https://github.com/cagrell/gp_plotly
dir_GPPlotly = "gp_plotly/"
sys.path.append(dir_GPPlotly)

# Path to the constrained GP moule
# can be downloaded at https://github.com/cagrell/gp_constr
dir_gp_constr = "gp_constr/"
sys.path.append(dir_gp_constr)

# Import
from GPPlotly.plottingfunctions import PlotGP2d, add_traces_to_fig
from GPConstr.model import GPmodel, Constraint
from GPConstr.kern import kernel_RBF

### Setup notebook ###
pltlyoff.init_notebook_mode(connected=True)
print("Python version", sys.version)

# %% [markdown]
# ## 1. Define function for generating synthetic test/training data
#


# %%
# Function to emulate/estimate
def fun(x):
    return (np.arctan(20 * x - 10) - np.arctan(-10)) / 3


# %% [markdown]
# ## 2. Regression with Gaussian noise

# %% [markdown]
# ### 2.1. Generate synthetic training data

# %%
# Design data - with noise
n = 50
noise_std = 0.2
x_design = np.random.uniform(0.1, 0.8, n)
y_design = fun(x_design) + np.random.normal(0, noise_std, n)

# For plotting
x_test = np.linspace(0, 1, 500)
y_true = fun(x_test)

# %% [markdown]
# ### 2.2. Define GP model (without constraints)

# %%
# Set up model
ker = kernel_RBF(variance=0.5, lengthscale=[0.1])
model = GPmodel(kernel=ker, likelihood=1, mean=0)

# Add the training data
model.X_training = x_design.reshape(-1, 1)
model.Y_training = y_design

# %%
# Optimize hyperparameters
model.optimize(include_constraint=False, fix_likelihood=False)
print(model)

# %%
# Plot unconstrained GP
mean_unconstr, cov_unconstr = model.calc_posterior_unconstrained(
    x_test.reshape(-1, 1), full_cov=True
)
mean_unconstr = np.array(mean_unconstr).flatten()
var_unconstr = np.diagonal(cov_unconstr)

num_samples = 30
show_samplepaths = True
samplepaths_unconstr = []
if show_samplepaths:
    samplepaths_unconstr = np.random.multivariate_normal(
        mean_unconstr, cov_unconstr, num_samples
    ).T

fig_unconstr_1 = PlotGP2d(
    x_mean=x_test,
    mean=mean_unconstr,
    var=var_unconstr,
    x_obs=model.X_training[:, 0],
    y_obs=model.Y_training,
    num_std=1.2815,
    x_true=x_test,
    y_true=y_true,
    samplepaths=samplepaths_unconstr,
    title="Unconstrained GP",
    xrange=[0, 1],
    yrange=[-1.7, 1.7],
    smoothing=True,
)

pltlyoff.iplot(fig_unconstr_1, filename="")

# %%
# Show plot as static image
Image(pio.to_image(fig_unconstr_1, width=1000, height=500, scale=1, format="png"))

# %% [markdown]
# ### 2.2.3. Include constraints (boundedness and monotonicity)


# %%
# Helper functions for constraints
def constant_function(val):
    """Return the constant function"""

    def fun(x):
        return np.array([val] * x.shape[0])

    return fun


def fun_UB(x):
    """Upper bound function"""
    return np.log(30 * x.flatten() + 1) / 3 + 0.1


# %%
# Define constraints for bounding the function and its derivative
constr_bounded = Constraint(LB=constant_function(0), UB=fun_UB)
constr_deriv = Constraint(LB=constant_function(0), UB=constant_function(float("Inf")))

# %%
# Add constraints to model
model.constr_bounded = constr_bounded
model.constr_deriv = [
    constr_deriv
]  # Add list of constraints for multi-dimensional functions
model.constr_likelihood = 1e-6

print(model)

# %%
# Search for a suitable set of virtual observation locations where the constraint is imposed
df, num_pts, pc_min = model.find_XV_subop(
    bounds=[(0.001, 1)], p_target=0.9, i_range=[0, 1]
)

# %%
# Plot model with both constraints

mean, var, perc, mode, samples, times = model.calc_posterior_constrained(
    x_test.reshape(-1, 1),
    compute_mode=False,
    num_samples=10000,
    save_samples=30,
    algorithm="minimax_tilting",
    resample=False,
)

mean = np.array(mean).flatten()
p_lower = perc[0]
median = perc[1]
p_upper = perc[2]
p_label = "[p{}, p{}]".format(10, 90)

samplepaths_Z = np.array(samples)

fig_both = PlotGP2d(
    x_mean=x_test,
    mean=mean,
    x_obs=model.X_training[:, 0],
    y_obs=model.Y_training,
    p_lower=p_lower,
    p_upper=p_upper,
    p_label=p_label,
    samplepaths=samplepaths_Z,
    x_true=x_test,
    y_true=y_true,
    title="Both constraints",
    xrange=[0, 1],
    yrange=[-1.7, 1.7],
    smoothing=True,
)

trace_XV_bounded = go.Scatter(
    x=model.constr_bounded.Xv.flatten(),
    y=np.zeros(model.constr_bounded.Xv.shape[0]),
    mode="markers",
    name="Xv - boundedness",
    marker=dict(symbol="line-ns-open", color=("rgb(0, 0, 0)")),
)
trace_XV_mon = go.Scatter(
    x=model.constr_deriv[0].Xv.flatten(),
    y=np.zeros(model.constr_deriv[0].Xv.shape[0]),
    mode="markers",
    name="Xv - monotonicity",
    marker=dict(symbol="x-thin-open", color=("rgb(0, 0, 0)")),
)
trace_UB = go.Scatter(
    x=x_test,
    y=model.constr_bounded.UB(x_test),
    mode="lines",
    name="Upper bound",
    line=dict(color=("rgb(0, 0, 0)"), shape="spline", width=1),
)
trace_LB = go.Scatter(
    x=x_test,
    y=model.constr_bounded.LB(x_test),
    mode="lines",
    name="Lower bound",
    line=dict(color=("rgb(0, 0, 0)"), shape="spline", width=1),
)

fig_both = add_traces_to_fig(
    fig_both, [trace_UB, trace_LB, trace_XV_bounded, trace_XV_mon]
)

pltlyoff.iplot(fig_both, filename="")


# %%
# Optimize constrained
# model.optimize(include_constraint = True, fix_likelihood = False)
# print(model)

# %%
# Show plot as static image
Image(pio.to_image(fig_both, width=1000, height=500, scale=1, format="png"))

# %% [markdown]
# #### Show all plots together - interactive

# %%
pltlyoff.iplot(fig_unconstr_1, filename="")
pltlyoff.iplot(fig_both, filename="")

# %% [markdown]
# #### Show all plots together - static

# %%
Image(pio.to_image(fig_unconstr_1, width=1000, height=500, scale=1, format="png"))

# %%
Image(pio.to_image(fig_both, width=1000, height=500, scale=1, format="png"))
