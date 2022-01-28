import numpy as np
import os
import pickle
from casbah import cgm as cas_cgm

import matplotlib.pyplot as plt
import cgmsquared.clustering as cgm2_cluster

from scipy.optimize import minimize


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import Table, vstack
from casbah import cgm as cas_cgm

from cgmsquared import clustering as cgm2_cluster
from cgmsquared import load_cgmsquared
import cgmsquared.clustering2 as c2

from astropy.stats import binom_conf_interval
import glob

import casbah.plotting as cplt

import scipy.special


import emcee
import corner

# globals
# clight = 2.9979246e5
clight = const.c.to(u.km / u.s).value


def phit_1halo(theta, z, rho_com, mass, massidx, Hz, dv, vmax=500.0):

    m0 = np.array([10 ** 9, 10 ** 9.5, 10 ** 10])

    if mass is None:
        print("If you want to model the mass you need to supply a mass array.")
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # use the analytic solution to chi_perp
    r0_mass = r0 * (mass / m0[massidx]) ** (beta[massidx])
    chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)

    # there should be the two halo term power law with "fixed" parameters from Tejos+2014
    m0_2h = 10 ** 9.5
    r0_mass_2 = r0_2 * (mass / m0_2h) ** (beta_2)
    chi_i_2halo = c2.chi_perp_analytic(r0_mass_2, gamma_2, rho_com, z, Hz, dv)

    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)

    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3

    rate_of_incidence = (1 + chi_i) * mean_dN_dz

    prob_miss = np.exp(-rate_of_incidence)
    prob_hit = 1 - prob_miss
    return prob_hit


def phit_2halo(
    theta, z, rho_com, mass, massidx, Hz, dv, vmax=500.0, gamma_2halo_fixed=1.7
):

    m0 = np.array([10 ** 9, 10 ** 9.5, 10 ** 10])

    if mass is None:
        print("If you want to model the mass you need to supply a mass array.")
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # use the analytic solution to chi_perp
    r0_mass = r0 * (mass / m0[massidx]) ** (beta[massidx])
    chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)

    # there should be the two halo term power law with "fixed" parameters from Tejos+2014
    m0_2h = 10 ** 9.5
    r0_mass_2 = r0_2 * (mass / m0_2h) ** (beta_2)
    chi_i_2halo = c2.chi_perp_analytic(r0_mass_2, gamma_2, rho_com, z, Hz, dv)
    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)

    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3

    rate_of_incidence = (1 + chi_i_2halo) * mean_dN_dz

    prob_miss = np.exp(-rate_of_incidence)
    prob_hit = 1 - prob_miss
    return prob_hit


def phit_2halo_only(
    theta, z, rho_com, mass, massidx, Hz, dv, vmax=500.0, gamma_2halo_fixed=1.7
):

    m0 = np.array([10 ** 9, 10 ** 9.5, 10 ** 10])

    if mass is None:
        print("If you want to model the mass you need to supply a mass array.")
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # use the analytic solution to chi_perp
    r0_mass = r0 * (mass / m0[massidx]) ** (beta[massidx])
    chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)

    # there should be the two halo term power law with "fixed" parameters from Tejos+2014
    m0_2h = 10 ** 9.5
    r0_mass_2 = r0_2 * (mass / m0_2h) ** (beta_2)
    chi_i_2halo = c2.chi_perp_analytic(r0_mass_2, gamma_2, rho_com, z, Hz, dv)

    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)

    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3

    rate_of_incidence = (1 + chi_i_2halo) * mean_dN_dz

    prob_miss = np.exp(-rate_of_incidence)
    prob_hit = 1 - prob_miss
    return prob_hit


def phit_2_power_law(
    theta, z, rho_com, mass, massidx, Hz, dv, vmax=500.0, gamma_2halo_fixed=1.7
):

    m0 = np.array([10 ** 9, 10 ** 9.5, 10 ** 10])

    if mass is None:
        print("If you want to model the mass you need to supply a mass array.")
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # use the analytic solution to chi_perp
    r0_mass = r0 * (mass / m0[massidx]) ** (beta[massidx])
    chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)

    # there should be the two halo term power law with "fixed" parameters from Tejos+2014
    m0_2h = 10 ** 9.5
    r0_mass_2 = r0_2 * (mass / m0_2h) ** (beta_2)
    chi_i_2halo = c2.chi_perp_analytic(r0_mass_2, gamma_2, rho_com, z, Hz, dv)

    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)

    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3

    rate_of_incidence = (1 + (chi_i + chi_i_2halo)) * mean_dN_dz

    prob_miss = np.exp(-rate_of_incidence)
    prob_hit = 1 - prob_miss
    return prob_hit


def phit_2_power_law_3betas(theta, z, rho_com, mass, massidx, Hz, dv, vmax=500):
    # theta, z, rho_com, mass, massidx, Hz, dv, vmax=vmax
    # fix m0
    m0 = np.array([10 ** 9, 10 ** 9.5, 10 ** 10])

    if mass is None:
        print("If you want to model the mass you need to supply a mass array.")
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # use the analytic solution to chi_perp
    r0_mass = r0 * (mass / m0[massidx]) ** (beta[massidx])
    chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)

    # there should be the two halo term power law with "fixed" parameters from Tejos+2014
    m0_2h = 10 ** 9.5
    r0_mass_2 = r0_2 * (mass / m0_2h) ** (beta_2)
    chi_i_2halo = c2.chi_perp_analytic(r0_mass_2, gamma_2, rho_com, z, Hz, dv)

    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)

    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3

    rate_of_incidence = (1 + (chi_i + chi_i_2halo)) * mean_dN_dz

    prob_miss = np.exp(-rate_of_incidence)
    prob_hit = 1 - prob_miss
    return prob_hit


def log_likelihood_2_power_law(
    theta, z, rho_com, mass, massidx, hits, misses, Hz, dv, vmax=500.0, gamma_2halo=None
):

    prob_hit = phit_2_power_law_3betas(
        theta, z, rho_com, mass, massidx, Hz, dv, vmax=500.0
    )

    # artifically inflating the variance.
    prob_hit = np.clip(prob_hit, 0.01, 0.99)

    prob_miss = 1 - prob_hit

    log_prob_hits = np.log(prob_hit[hits])
    log_prob_miss = np.log(prob_miss[misses])

    sum_log_prob_miss = np.sum(log_prob_miss)
    sum_log_prob_hits = np.sum(log_prob_hits)

    llikelihood = sum_log_prob_hits + sum_log_prob_miss
    return llikelihood


nll = lambda *args: -log_likelihood_2_power_law(*args)


def log_prior(theta):
    (
        r0,
        r0_2,
        gamma,
        gamma_2,
        beta0,
        beta1,
        beta2,
        beta_2,
        dndz_index,
        dndz_coeff,
    ) = theta

    beta = np.array([beta0, beta1, beta2])

    # flat prior on r0, gaussian prior on gamma around 1.6
    if (r0 < 0) or (r0 > 5) or (r0_2 < 0) or (r0_2 > 10):
        return -np.inf
    if (gamma < 2) or (gamma > 10) or (gamma_2 < 0) or (gamma_2 > 10):
        return -np.inf
    if (beta.any() < -1) or (beta.any() > 5) or (beta_2 < -1):
        return -np.inf
    if (dndz_index < -3) or (dndz_index > 3) or (dndz_coeff < 0) or (dndz_coeff > 40):
        return -np.inf

    sig = 1.0
    ln_prior = -0.5 * ((gamma - 6) ** 2 / (sig) ** 2)
    ln_prior = -0.5 * ((gamma_2 - 1.7) ** 2 / (0.1) ** 2)  # tejos 2014
    ln_prior += -0.5 * ((r0 - 1) ** 2 / (sig) ** 2)
    ln_prior += -0.5 * ((r0_2 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
    # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)
    ln_prior += np.sum(-0.5 * ((beta - 0.5) ** 2 / (sig) ** 2))
    ln_prior += -0.5 * ((beta_2 - 0.5) ** 2 / (sig) ** 2)
    ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
    ln_prior += -0.5 * (
        (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
    ) - np.log(
        dndz_coeff
    )  # kim+

    return ln_prior


def log_probability(theta, *args):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp - nll(theta, *args)
