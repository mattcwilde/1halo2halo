""" this is where the max_likelihood_est func etc will go"""
import emcee
import numpy as np


def max_likelihood_est(model, initial, bounds):
    from scipy.optimize import minimize

    # nll = lambda *args: model.neg_log_likelihood(args)
    soln = minimize(model.neg_log_likelihood, x0=initial, bounds=bounds)
    return soln


def posterior_sampler(soln, log_probability, nsteps=10000):
    pos = soln.x + 1e-4 * np.random.randn(2 * soln.x.shape[0], soln.x.shape[0])
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    sampler.run_mcmc(pos, nsteps, progress=True)
    return sampler

