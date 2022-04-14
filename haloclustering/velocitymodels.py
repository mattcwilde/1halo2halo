import cgmsquared.clustering2 as c2
from cgmsquared import clustering as cgm2_cluster
from astropy.cosmology import Planck15 as cosmo
import numpy as np


class velocityModelSingle(object):
    def __init__(self, df) -> None:
        # galaxy data
        self.rho_com = df.rho.values / 1000
        self.z = df.z.values
        self.Hz = cosmo.H(self.z).value
        outcomes = df.filter(like="HM_").drop(columns="HM_0_500")
        self.hits = outcomes.values == "hit"
        self.misses = outcomes.values == "miss"
        self.rvir = df.rvir.values / 1000  # convert to Mpc

    def chi_perp_vlist(self, r0, gamma, dvlist):
        chi_list = []
        chidv_last = 0
        dvlast = 0
        for dv in dvlist:
            chi = c2.chi_perp_analytic(r0, gamma, self.rho_com, self.z, self.Hz, dv)
            chidv = chi * dv
            chi_list.append((chidv - chidv_last) / (dv - dvlast))
            chidv_last = chidv
            dvlast = dv

        chi_arr = np.array(chi_list)

        return chi_arr.T

    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0_2, gamma_2, dndz_index, dndz_coeff = params

        try:
            self.r0_2 = r0_2[:, None]
            self.gamma_2 = gamma_2[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
            self.params = params[:, None]
        except IndexError:
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff
            self.params = params

    def mean_dNdz(self, dndz_coeff, dndz_index, vmax=100.0):
        # dN_dz for HI with logNHI > 14
        ion_lz = cgm2_cluster.hi_lz(
            self.z, danforth=False, gamma=dndz_index, coeff=dndz_coeff
        )
        clight = 299792.458
        # mean number of absorbers along line of sight in dz window
        dz = (1 + self.z) * (2 * vmax / clight)
        mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3
        return mean_dN_dz

    def phit(self, chi_arr, mean_dNdz):
        rate_of_incidence = (1 + chi_arr) * mean_dNdz[:, None]
        prob_miss = np.exp(-rate_of_incidence)
        prob_hit = 1 - prob_miss
        prob_hit = np.clip(prob_hit, 0.00001, 0.99)
        return prob_hit

    def log_likelihood(self, params):
        r0, gamma, dndz_coeff, dndz_index = params

        chi_arr = self.chi_perp_vlist(r0, gamma, self.dvlist)
        meandNdz = self.mean_dNdz(dndz_coeff, dndz_index)
        prob_hit = self.phit(chi_arr, meandNdz)
        prob_miss = 1 - prob_hit
        log_prob_hits = np.log(prob_hit[self.hits])
        log_prob_miss = np.log(prob_miss[self.misses])

        sum_log_prob_miss = np.sum(log_prob_miss)
        sum_log_prob_hits = np.sum(log_prob_hits)

        llikelihood = sum_log_prob_hits + sum_log_prob_miss
        return llikelihood

    def log_prior(self, params):
        """the Bayesian prior. Will change with each model based on which params are 
        important to the model. 

        Returns:
            ln_prior (float): natural log of the prior
        """
        r0, gamma, dndz_coeff, dndz_index = params

        # flat prior on r0, gaussian prior on gamma around 1.6

        if (r0 < 0) or (r0 > 10):
            return -np.inf
        if (gamma < 0) or (gamma > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        ln_prior = -0.5 * ((gamma - 1.7) ** 2 / (0.1) ** 2)  # tejos 2014
        ln_prior += -0.5 * ((r0 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
        # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)
        ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
        ln_prior += -0.5 * (
            (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
        ) - np.log(
            dndz_coeff
        )  # kim+ # log-normal has a 1/x

        return ln_prior

    def log_probability(self, params):
        lp = self.log_prior(params)

        if not np.isfinite(lp):
            return -np.inf
        logprob = lp + self.log_likelihood(params)
        return logprob


class velocityModel1h2h(object):
    def __init__(self, df, dvlist, m0=10 ** 9.5, mockdata=False, z=None) -> None:
        if mockdata:
            self.m0 = m0
            self.dvlist = dvlist
            if z is not None:
                self.z = z
            else:
                z = np.array([0.3])
            self.rvir = 1.0
            self.rho_com = df.rho.sort_values().values / 1000
            self.Hz = cosmo.H(self.z).value
            self.hits = np.ones(len(dvlist))
            self.misses = np.zeros(len(dvlist))
            self.rvir = 1.0
            self.mass = self.m0
        else:
            # galaxy data
            self.rho_com = df.rho.values / 1000
            self.z = df.z.values
            self.Hz = cosmo.H(self.z).value
            outcomes = df.filter(like="HM_").drop(columns="HM_0_500")
            self.hits = outcomes.values == "hit"
            self.misses = outcomes.values == "miss"
            self.rvir = df.rvir.values / 1000  # convert to Mpc
            self.dvlist = dvlist
            self.mass = df.mstars.values
            self.m0 = m0

    def set_params(self, params) -> None:
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0, gamma, r0_2, gamma_2, beta, beta2h, dndz_index, dndz_coeff = params
        try:
            self.r0 = r0[:, None]
            self.gamma = gamma[:, None]
            self.r0_2 = r0_2[:, None]
            self.gamma_2 = gamma_2[:, None]
            self.beta = beta[:, None]
            self.beta2h = beta2h[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
            self.params = params[:, None]
        except:
            self.r0 = r0
            self.gamma = gamma
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.beta = beta
            self.beta2h = beta2h
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff
            self.params = params

    def r0func_1h(self):
        r0_mass = self.r0 * (self.mass / self.m0) ** (self.beta)
        return r0_mass

    def r0func_2h(self):
        r0_mass = self.r0_2 * (self.mass / self.m0) ** (self.beta2h)
        return r0_mass

    def chi_perp(self, r0, gamma):
        chi_list = []
        chidv_last = 0
        dvlast = 0
        for dv in self.dvlist:
            chi = c2.chi_perp_analytic(r0, gamma, self.rho_com, self.z, self.Hz, dv)
            chidv = chi * dv
            chi_list.append((chidv - chidv_last) / (dv - dvlast))
            chidv_last = chidv
            dvlast = dv

        chi_arr = np.array(chi_list)

        return chi_arr.T

    def mean_dNdz(self, vmax=100.0):
        # dN_dz for HI with logNHI > 14
        ion_lz = cgm2_cluster.hi_lz(
            self.z, danforth=False, gamma=self.dndz_index, coeff=self.dndz_coeff
        )
        clight = 299792.458
        # mean number of absorbers along line of sight in dz window
        dz = 2 * (1 + self.z) * (vmax / clight)
        mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3
        return mean_dN_dz

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        sum_of_chi = chi_perp1 + chi_perp2

        rate_of_incidence = (1 + sum_of_chi) * self.mean_dNdz()[:, None]
        prob_miss = np.exp(-rate_of_incidence)
        prob_hit = 1 - prob_miss
        prob_hit = np.clip(prob_hit, 0.00001, 0.99)
        return prob_hit

    def log_prior(self):
        (r0, gamma, r0_2, gamma_2, beta, beta2h, dndz_index, dndz_coeff) = self.params

        # flat prior on r0, gaussian prior on gamma around 1.6
        if (r0 < 0) or (r0 > 10) or (r0_2 < 0) or (r0_2 > 10):
            return -np.inf
        if (gamma < 1.9) or (gamma > 10) or (gamma_2 < 1) or (gamma_2 > 10):
            return -np.inf
        if (beta < -3) or (beta > 10) or (beta2h < -3) or (beta2h > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        sig = 1.0
        # TODO: check other priors for gamma 1-halo
        # ln_prior = -0.5 * ((gamma - 2) ** 2 / (sig) ** 2)
        ln_prior = -0.5 * ((gamma_2 - 1.7) ** 2 / (0.1) ** 2)  # tejos 2014

        # ln_prior += -0.5 * ((r0 - 1) ** 2 / (sig) ** 2)
        ln_prior += -0.5 * ((r0_2 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
        # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)

        ln_prior += -0.5 * ((beta - 1 / 8) ** 2 / (sig) ** 2)  # mirror rvir slope
        ln_prior += -0.5 * ((beta2h - 0) ** 2 / (sig) ** 2)  # no mass dependence?
        ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
        ln_prior += -0.5 * (
            (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
        ) - np.log(
            dndz_coeff
        )  # kim+

        return ln_prior

    def log_likelihood(self):

        prob_hit = self.phit_sum()
        prob_miss = 1 - prob_hit
        log_prob_hits = np.log(prob_hit[self.hits])
        log_prob_miss = np.log(prob_miss[self.misses])

        sum_log_prob_miss = np.sum(log_prob_miss)
        sum_log_prob_hits = np.sum(log_prob_hits)

        llikelihood = sum_log_prob_hits + sum_log_prob_miss
        return llikelihood

    def neg_log_likelihood(self, params=None):
        if params is not None:
            self.set_params(params)
        return -self.log_likelihood()

    def log_probability(self, params=None):
        if params is not None:
            self.set_params(params)
        lp = self.log_prior()

        if not np.isfinite(lp):
            return -np.inf
        logprob = lp + self.log_likelihood()
        return logprob
