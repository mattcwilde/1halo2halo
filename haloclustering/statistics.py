import pickle
import numpy as np
from cgmsquared import clustering as cgm2_cluster
from cgmsquared import clustering2 as c2
import pandas as pd
import scipy.special as sc


def get_sampler(pkl_file):
    infile = open(pkl_file, "rb")
    sampler = pickle.load(infile)
    infile.close()
    return sampler


def anti_derivative_full(r0, gamma, r_parallel, r_perpendicular):
    """
    This is the analytic solution to the integrand.

    see the solution at https://www.wolframalpha.com/input/?i=integral+%28sqrt%28x%5E2+%2B+r%5E2%29%2Fr0%29%5E-g


    Args:
        r0: scale length in comoving Mpc
        gamma: power law term.
        r_parallel: the integrand
        r_perpendicular: the rho_comoving_mpc measurements from the galaxies

    Returns:
        full: the anti-derivative

    """
    x = r_parallel
    r = r_perpendicular
    g = gamma
    full = (
        (x * (np.sqrt(r ** 2 + x ** 2) / r0) ** (-g))
        * (1 + x ** 2 / r ** 2) ** (g / 2)
        * sc.hyp2f1(1 / 2, g / 2, 3 / 2, -(x ** 2) / r ** 2)
    )
    return full


def integral_1h(rho, A, sigma, s_eval):
    part1 = np.sqrt(np.pi / 2) * sigma * np.exp(-(rho ** 2) / (2 * sigma ** 2))
    part2 = sc.erf(s_eval / (np.sqrt(2) * sigma))
    return A * part1 * part2


def integral_2h(rho, s_max, s_eval, r0, gamma):
    part1 = anti_derivative_full(r0, gamma, s_max, rho)  # Mpc
    part2 = anti_derivative_full(r0, gamma, s_eval, rho)  # Mpc
    return part1 - part2


def chi_perp_exc(r0, gamma, A, sigma, r_cross, rho_gal_com, z_gal, Hz, vmax):
    dv = vmax  # km/s
    a = 1 / (1 + z_gal)
    norm_const = a * Hz / (2 * dv)  # 1/Mpc
    s_max = dv / (a * Hz)  # Mpc
    s_cross = np.sqrt(np.maximum(r_cross ** 2 - rho_gal_com ** 2, 0))
    s_eval = np.minimum(s_max, s_cross)
    integrand = integral_1h(rho_gal_com, A, sigma, s_eval) + integral_2h(
        rho_gal_com, s_max, s_eval, r0, gamma
    )
    chi = norm_const * integrand
    return chi


def _choose_rc(rc, A, r0, gamma):
    """ Choose which crossing to use.

    Args:
        rc (_type_): _description_
        A (_type_): _description_
        r0 (_type_): _description_
        gamma (_type_): _description_

    Returns:
        _type_: _description_
    """
    deriv = 2 * np.log(A) + gamma * (2 * np.log(rc / r0) - 1)
    if np.all(deriv > 0):
        return rc
    else:
        return None


def _calc_sig(rc, A, r0, gamma):
    r = _choose_rc(rc, A, r0, gamma)
    if r is None:
        return None
    else:
        sig = 0.5 * r ** 2 / (np.log(A) + gamma * (np.log(r / r0)))
        return np.sqrt(sig)


def mean_dNdz(z, dndz_index, dndz_coeff, vmax=500.0):
    # dN_dz for HI with logNHI > 14
    ion_lz = cgm2_cluster.hi_lz(z, danforth=False, gamma=dndz_index, coeff=dndz_coeff)
    clight = 299792.458
    # mean number of absorbers along line of sight in dz window
    dz = 2 * (1 + z) * (vmax / clight)
    mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3
    return mean_dN_dz


def r_cross_beta(r_cross, beta1h, mass, m0=10 ** 9.5):
    r_cross = r_cross * (mass / m0) ** beta1h
    return r_cross


def r0_beta(r0, beta2h, mass, m0=10 ** 9.5):
    r0 = r0 * (mass / m0) ** beta2h
    return r0


def phit_exc(
    r0,
    gamma,
    beta2h,
    A,
    r_cross,
    beta1h,
    dndz_index,
    dndz_coeff,
    rho_gal_com,
    z_gal,
    Hz,
    vmax,
    mass,
):
    r_cross = r_cross_beta(r_cross, beta1h, mass, m0=10 ** 9.5)
    r0 = r0_beta(r0, beta2h, mass, m0=10 ** 9.5)
    sigma = _calc_sig(r_cross, A, r0, gamma)
    if sigma is None:
        return None
    chi = chi_perp_exc(r0, gamma, A, sigma, r_cross, rho_gal_com, z_gal, Hz, vmax)
    dndz = mean_dNdz(z_gal, dndz_index, dndz_coeff)
    phit = 1 - np.exp(-(1 + chi) * dndz)
    phit = np.clip(phit, 0.00001, 0.99)
    return phit


def log_likelihood_exc(
    r0,
    gamma,
    beta2h,
    A,
    r_cross,
    beta1h,
    dndz_index,
    dndz_coeff,
    rho_gal_com,
    z_gal,
    Hz,
    vmax,
    mass,
    hits,
    misses,
):
    prob_hit = phit_exc(
        r0,
        gamma,
        beta2h,
        A,
        r_cross,
        beta1h,
        dndz_index,
        dndz_coeff,
        rho_gal_com,
        z_gal,
        Hz,
        vmax,
        mass,
    )
    if prob_hit is None:
        return -np.inf, np.full(mass.shape, np.nan)
    else:
        prob_miss = 1 - prob_hit
        log_prob_hits = np.log(prob_hit[hits])
        log_prob_miss = np.log(prob_miss[misses])

        sum_log_prob_miss = np.sum(log_prob_miss)
        sum_log_prob_hits = np.sum(log_prob_hits)

        llikelihood = sum_log_prob_hits + sum_log_prob_miss
        return llikelihood, prob_hit


def chi_perp_2h(r0, gamma, rho_gal_com, z_gal, Hz, vmax):
    """compute the integral of the clustering function along the line of site. 
    Use the analytic solution. works for either a single valued r0, gamma or broadcasts
    correctly with r0 ~ Rvir of each galaxy. 

    Args:
        r0 (float, ndarray): scaling factor for the clustering power law. can be a float or an array
            with length matching that of the number of galaxies.
        gamma (float): clustering power law index. 

    Returns:
        ndarray: chi_perp array with length equal to number of galaxies.
    """
    chi_i = c2.chi_perp_analytic(r0, gamma, rho_gal_com, z_gal, Hz, vmax)
    return chi_i


def phit_2h(
    r0, gamma, beta2h, dndz_index, dndz_coeff, rho_gal_com, z_gal, Hz, vmax, mass
):

    r0 = r0_beta(r0, beta2h, mass, m0=10 ** 9.5)

    chi = chi_perp_2h(r0, gamma, rho_gal_com, z_gal, Hz, vmax)
    dndz = mean_dNdz(z_gal, dndz_index, dndz_coeff)
    phit = 1 - np.exp(-(1 + chi) * dndz)
    phit = np.clip(phit, 0.00001, 0.99)
    return phit


def log_likelihood_2h(
    r0,
    gamma,
    beta2h,
    dndz_index,
    dndz_coeff,
    rho_gal_com,
    z_gal,
    Hz,
    vmax,
    mass,
    hits,
    misses,
):
    print(r0, gamma, beta2h)
    prob_hit = phit_2h(
        r0, gamma, beta2h, dndz_index, dndz_coeff, rho_gal_com, z_gal, Hz, vmax, mass
    )
    if prob_hit is None:
        return -np.inf, np.full(mass.shape, np.nan)
    else:
        prob_miss = 1 - prob_hit
        log_prob_hits = np.log(prob_hit[hits])
        log_prob_miss = np.log(prob_miss[misses])

        sum_log_prob_miss = np.sum(log_prob_miss)
        sum_log_prob_hits = np.sum(log_prob_hits)

        llikelihood = sum_log_prob_hits + sum_log_prob_miss
        return llikelihood, prob_hit


def bayes_inf_criterion(likelihood, num_params, num_data):
    bic = -2 * likelihood + num_params * np.log(num_data)
    return bic
