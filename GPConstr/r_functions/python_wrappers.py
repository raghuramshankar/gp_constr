# %%
### Python wrappers for R functions ###
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
import numpy as np
import os

# install r packages if not already installed
packageNames = ["tmvtnorm"]
packnames_to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
if len(packnames_to_install) > 0:
    utils.install_packages(StrVector(packnames_to_install))

# Import custom files and define functions using R
tmvtnorm = rpackages.importr("tmvtnorm")
r_rtmvnorm = robjects.r["rtmvnorm"]
r_pmvnorm = robjects.r["pmvnorm"]
r_mtmvnorm = robjects.r["mtmvnorm"]

# Print R version
print("Running R from rpy2: {}".format(robjects.r("R.Version()$version.string")[0]))


def param_py_to_r(mu, sigma, a, b):
    """Convert to r objects"""

    # R vector
    r_mu = robjects.FloatVector(mu)
    r_a = robjects.FloatVector(a)
    r_b = robjects.FloatVector(b)

    # R matrix
    sigma_flat = sigma.flatten().tolist()[0]
    r_sigma = robjects.r["matrix"](
        robjects.FloatVector(sigma_flat), nrow=sigma.shape[0]
    )

    return r_mu, r_sigma, r_a, r_b


def rtmvnorm(n, mu, sigma, a, b, algorithm="gibbs"):
    """
    Create n samples from truncated multivariate normal with
    mean = mu and covariance matrix = sigma

    a = lower bound, b = upper bound

    n : integer
    mu : numpy array
    sigma : numpy matrix
    a : numpy array
    b : numpy array
    H (optional) : precision matrix (inverse of sigma) <-- removed this
    """

    # Convert to R objects
    r_mu, r_sigma, r_a, r_b = param_py_to_r(mu, sigma, a, b)

    # Run R function and cast to numpy array
    X = np.array(r_rtmvnorm(n, r_mu, r_sigma, r_a, r_b, algorithm))

    assert not np.isnan(np.min(X)), "rtmvnorm returns nan"

    return X


def pmvnorm(mu, sigma, a, b, algorithm, n=1e4):
    """Returns probability over rectangle given by [a, b]"""
    # Convert to R objects
    r_mu, r_sigma, r_a, r_b = param_py_to_r(mu, sigma, a, b)

    return np.array(r_pmvnorm(r_mu, r_sigma, r_a, r_b, algorithm, n))[0]


def mtmvnorm(mu, sigma, a, b):
    """Returns moments of truncated multivariate normal"""
    # Convert to R objects
    r_mu, r_sigma, r_a, r_b = param_py_to_r(mu, sigma, a, b)

    moments = r_mtmvnorm(r_mu, r_sigma, r_a, r_b)

    return np.array(moments[0]), np.array(moments[1])


def moments_from_samples(n, mu, sigma, a, b, algorithm="minimax_tilting"):
    """
    Estimate moments of truncated normal from n samples
    """
    samples = rtmvnorm(n, mu, sigma, a, b, algorithm)
    return samples.mean(axis=0), np.cov(samples, rowvar=False)
