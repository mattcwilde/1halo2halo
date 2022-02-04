import cgmsquared.clustering2 as c2
import numpy as np
from cgmsquared import clustering as cgm2_cluster


class Model:
    """This model of the covering fraction/probability of hitting logNHI > 10**14 is based on
    using r0 ~ Rvir. 

    This one takes in data that should be an array of values/arrays and optionally an m0. 

    Add more here eventually. 

    TODO: make a generic model for the regular case. 
    """

    def __init__(self, data) -> None:
        self.data = data
        # unpack the data variables
        (
            self.z,
            self.rho_com,
            self.mass,
            self.hits,
            self.misses,
            self.Hz,
            self.dv,
            self.rvir,
            self.do_anly,
        ) = self.data
        # check if an astropy table or just an array
        if self.rvir.dtype == np.float64:
            self.r0_rvir = self.rvir / 1000.0
        else:
            self.r0_rvir = self.rvir.data / 1000.0  # convert to Mpc ~ rho_com

    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0_coeff, gamma, r0_2, gamma_2, dndz_index, dndz_coeff = params
        # 1 halo and 2 halo parameters theta
        # r0_1halo is going to be the rvir
        self.r0_coeff = r0_coeff
        self.gamma = gamma
        self.r0_2 = r0_2
        self.gamma_2 = gamma_2
        self.dndz_index = dndz_index
        self.dndz_coeff = dndz_coeff

    def r0(self):
        return self.r0_rvir * self.r0_coeff

    def chi_perp(self, r0, gamma):
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
        chi_i = c2.chi_perp_analytic(r0, gamma, self.rho_com, self.z, self.Hz, self.dv)
        return chi_i

    def mean_dNdz(self, vmax=500.0):
        # dN_dz for HI with logNHI > 14
        ion_lz = cgm2_cluster.hi_lz(
            self.z, danforth=False, gamma=self.dndz_index, coeff=self.dndz_coeff
        )
        clight = 299792.458
        # mean number of absorbers along line of sight in dz window
        dz = 2 * (1 + self.z) * (vmax / clight)
        mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3
        return mean_dN_dz

    def _calc_prob(self, chi_perp):
        rate_of_incidence = (1 + chi_perp) * self.mean_dNdz()
        prob_miss = np.exp(-rate_of_incidence)
        prob_hit = 1 - prob_miss
        return prob_hit

    def phit_1halo(self):
        chi_perp1 = self.chi_perp(self.r0(), self.gamma)
        prob_hit = self._calc_prob(chi_perp1)
        return prob_hit

    def phit_2halo(self):
        chi_perp2 = self.chi_perp(self.r0_2, self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
        return prob_hit

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0(), self.gamma)
        chi_perp2 = self.chi_perp(self.r0_2, self.gamma_2)
        prob_hit = self._calc_prob(chi_perp1 + chi_perp2)
        return prob_hit

    def log_likelihood(self):

        prob_hit = self.phit_sum()

        # artifically inflating the variance.
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
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

    def log_prior(self):
        """the Bayesian prior. Will change with each model based on which params are 
        important to the model. 

        Returns:
            ln_prior (float): natural log of the prior
        """
        r0_coeff = self.r0_coeff
        r0_2 = self.r0_2
        gamma = self.gamma
        gamma_2 = self.gamma_2
        dndz_index = self.dndz_index
        dndz_coeff = self.dndz_coeff

        # flat prior on r0, gaussian prior on gamma around 1.6
        if (r0_coeff < 0) or (r0_coeff > 10):
            return -np.inf
        if (r0_2 < 0) or (r0_2 > 10):
            return -np.inf
        if (gamma < 2) or (gamma > 10) or (gamma_2 < 0) or (gamma_2 > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        sig = 1.0
        ln_prior = -0.5 * ((gamma - 6) ** 2 / (sig) ** 2)
        ln_prior = -0.5 * ((gamma_2 - 1.7) ** 2 / (0.1) ** 2)  # tejos 2014
        ln_prior += -0.5 * ((r0_2 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
        # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)
        ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
        ln_prior += -0.5 * (
            (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
        ) - np.log(
            dndz_coeff
        )  # kim+

        return ln_prior

    def log_probability(self, params=None):
        if params is not None:
            self.set_params(params)
        lp = self.log_prior()

        if not np.isfinite(lp):
            return -np.inf
        logprob = lp + self.log_likelihood()
        return logprob


class Model2h(Model):
    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0_2, gamma_2, dndz_index, dndz_coeff = params
        # 1 halo and 2 halo parameters theta
        # r0_1halo is going to be the rvir
        self.r0_2 = r0_2
        self.gamma_2 = gamma_2
        self.dndz_index = dndz_index
        self.dndz_coeff = dndz_coeff

    def log_likelihood(self):

        prob_hit = self.phit_2halo()

        # artifically inflating the variance.
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        prob_miss = 1 - prob_hit
        log_prob_hits = np.log(prob_hit[self.hits])
        log_prob_miss = np.log(prob_miss[self.misses])

        sum_log_prob_miss = np.sum(log_prob_miss)
        sum_log_prob_hits = np.sum(log_prob_hits)

        llikelihood = sum_log_prob_hits + sum_log_prob_miss
        return llikelihood

    def log_prior(self):
        """the Bayesian prior. Will change with each model based on which params are 
        important to the model. 

        Returns:
            ln_prior (float): natural log of the prior
        """
        r0_2 = self.r0_2
        gamma_2 = self.gamma_2
        dndz_index = self.dndz_index
        dndz_coeff = self.dndz_coeff

        # flat prior on r0, gaussian prior on gamma around 1.6

        if (r0_2 < 0) or (r0_2 > 10):
            return -np.inf
        if (gamma_2 < 0) or (gamma_2 > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        ln_prior = -0.5 * ((gamma_2 - 1.7) ** 2 / (0.1) ** 2)  # tejos 2014
        ln_prior += -0.5 * ((r0_2 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
        # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)
        ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
        ln_prior += -0.5 * (
            (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
        ) - np.log(
            dndz_coeff
        )  # kim+

        return ln_prior
