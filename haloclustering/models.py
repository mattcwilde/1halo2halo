import cgmsquared.clustering2 as c2
import numpy as np
from cgmsquared import clustering as cgm2_cluster
from scipy.special import erf


class Model:
    def __init__(self, data, m0=10 ** 9.5) -> None:

        self.data = data
        self.m0 = m0
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
            self.cgm_data_doanly,
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
        except IndexError:
            self.r0 = r0
            self.gamma = gamma
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.beta = beta
            self.beta2h = beta2h
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff
            self.params = params

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

    def r0func_1h(self):
        r0_mass = self.r0 * (self.mass / self.m0) ** (self.beta)
        return r0_mass

    def r0func_2h(self):
        r0_mass = self.r0_2 * (self.mass / self.m0) ** (self.beta2h)
        return r0_mass

    def phit_1halo(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        prob_hit = self._calc_prob(chi_perp1 - 1)  # removing radnom term from 1-halo
        # artifically inflating the variance.
        # based on numerics
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        return prob_hit

    def phit_2halo(self):
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
        # artifically inflating the variance.
        # based on numerics
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        return prob_hit

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        prob_hit = self._calc_prob(chi_perp1 + chi_perp2)
        # artifically inflating the variance.
        # based on numerics
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
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


class rvirModel(Model):
    """This model of the covering fraction/probability of hitting logNHI > 10**14 is based on
    using r0 ~ Rvir. 

    This one takes in data that should be an array of values/arrays and optionally an m0. 

    Add more here eventually. 

    """

    def __init__(self, data) -> None:
        super().__init__(data)

    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0_coeff, gamma, r0_2, gamma_2, dndz_index, dndz_coeff = params
        # 1 halo and 2 halo parameters theta
        # r0_1halo is going to be the rvir
        try:
            self.r0_coeff = r0_coeff[:, None]
            self.gamma = gamma[:, None]
            self.r0_2 = r0_2[:, None]
            self.gamma_2 = gamma_2[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
        except IndexError:
            self.r0_coeff = r0_coeff
            self.gamma = gamma
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff

    def r0func(self):
        return self.r0_rvir * self.r0_coeff

    def phit_1halo(self):
        chi_perp1 = self.chi_perp(self.r0func(), self.gamma)
        prob_hit = self._calc_prob(chi_perp1 - 1)
        return prob_hit

    def phit_2halo(self):
        chi_perp2 = self.chi_perp(self.r0_2, self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
        return prob_hit

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0func(), self.gamma)
        chi_perp2 = self.chi_perp(self.r0_2, self.gamma_2)
        prob_hit = self._calc_prob(chi_perp1 + chi_perp2)
        return prob_hit

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
        if (gamma < 0) or (gamma > 10) or (gamma_2 < 0) or (gamma_2 > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        sig = 1.0
        # ln_prior = -0.5 * ((gamma - 6) ** 2 / (sig) ** 2)
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


class Model2h(Model):
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

    def r0func_2h(self):
        r0_mass = self.r0_2
        return r0_mass

    def phit_sum(self):
        chi_perp2 = self.chi_perp(self.r0_2, self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
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
        )  # kim+ # log-normal has a 1/x

        return ln_prior


class ModelBetaMass(Model):
    def __init__(self, data, m0=10 ** 10.5) -> None:
        super().__init__(data)

        self.m0 = m0

    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """

        r0, gamma, r0_2, gamma_2, beta1, beta2, beta2h, dndz_index, dndz_coeff = params

        # 1 halo and 2 halo parameters theta
        # r0_1halo is going to be the rvir
        try:
            self.r0 = r0[:, None]
            self.gamma = gamma[:, None]
            self.r0_2 = r0_2[:, None]
            self.gamma_2 = gamma_2[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
            self.beta1h = np.array([beta1[:, None], beta2[:, None]])
            self.beta2h = beta2h[:, None]
            self.params = params[:, None]
        except IndexError:
            self.r0 = r0
            self.gamma = gamma
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff
            self.beta1h = np.array([beta1, beta2])
            self.beta2h = beta2h
            self.params = params

    def r0func(self):
        print("this isn't defined in this model")

    def r0func_1h(self):
        massidx = np.digitize(self.mass, [0, self.m0]) - 1
        r0_mass = self.r0 * (self.mass / self.m0) ** (self.beta1h[massidx])
        return r0_mass

    def r0func_2h(self):
        r0_mass = self.r0_2 * (self.mass / self.m0) ** (self.beta2h)
        return r0_mass

    def phit_1halo(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        prob_hit = self._calc_prob(chi_perp1 - 1)
        return prob_hit

    def phit_2halo(self):
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
        return prob_hit

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        prob_hit = self._calc_prob(chi_perp1 + chi_perp2)
        return prob_hit

    def log_prior(self):
        (
            r0,
            gamma,
            r0_2,
            gamma_2,
            beta1,
            beta2,
            beta2h,
            dndz_index,
            dndz_coeff,
        ) = self.params

        # flat prior on r0, gaussian prior on gamma around 1.6
        if (r0 < 0) or (r0 > 10) or (r0_2 < 0) or (r0_2 > 10):
            return -np.inf
        if (gamma < 0) or (gamma > 10) or (gamma_2 < 0) or (gamma_2 > 10):
            return -np.inf
        if (
            (beta1 < -3)
            or (beta1 > 10)
            or (beta2 < -3)
            or (beta2 > 10)
            or (beta2h < -3)
            or (beta2h > 10)
        ):
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
        ln_prior += -0.5 * ((r0 - 1) ** 2 / (sig) ** 2)
        ln_prior += -0.5 * ((r0_2 - 3.8) ** 2 / (0.3) ** 2)  # tejos 2014
        # ln_prior += -0.5*((beta - 0.5)**2/(sig)**2)

        # ln_prior += -0.5 * ((beta1 - 1 / 8) ** 2 / (sig) ** 2)
        # ln_prior += -0.5 * ((beta2 - 0.8) ** 2 / (sig) ** 2)
        # ln_prior += -0.5 * ((beta2h - 1 / 8) ** 2 / (sig) ** 2)
        ln_prior += -0.5 * ((dndz_index - 0.97) ** 2 / (0.87) ** 2)  # kim+
        ln_prior += -0.5 * (
            (np.log(dndz_coeff) - np.log(10) * 1.25) ** 2 / (np.log(10) * 0.11) ** 2
        ) - np.log(
            dndz_coeff
        )  # kim+

        return ln_prior


class Model2hBeta(Model):
    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0, gamma, beta, dndz_index, dndz_coeff = params

        try:
            self.r0 = r0[:, None]
            self.gamma = gamma[:, None]
            self.beta = beta[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
            self.params = params[:, None]
        except IndexError:
            self.r0 = r0
            self.gamma = gamma
            self.beta = beta
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff
            self.params = params

    def r0func_1h(self):
        r0_mass = self.r0 * (self.mass / self.m0) ** self.beta
        return r0_mass

    def phit_sum(self):
        chi_perp1 = self.chi_perp(self.r0func_1h(), self.gamma)
        prob_hit = self._calc_prob(chi_perp1)
        # artifically inflating the variance.
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        return prob_hit

    def log_prior(self):
        """the Bayesian prior. Will change with each model based on which params are 
        important to the model. 

        Returns:
            ln_prior (float): natural log of the prior
        """
        r0 = self.r0
        gamma = self.gamma
        beta = self.beta
        dndz_index = self.dndz_index
        dndz_coeff = self.dndz_coeff

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
        if (beta < -3) or (beta > 10):
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


class expModelRvir(rvirModel):
    def __init__(self, data) -> None:
        super().__init__(data)

    def set_params(self, params):
        """set the params specific to each model.

        Args:
            params (array): array of parameter values
        """
        r0_coeff, r0_2, gamma_2, dndz_index, dndz_coeff = params
        # 1 halo and 2 halo parameters theta
        # r0_1halo is going to be the rvir
        try:
            self.r0_coeff = r0_coeff[:, None]
            self.r0_2 = r0_2[:, None]
            self.gamma_2 = gamma_2[:, None]
            self.dndz_index = dndz_index[:, None]
            self.dndz_coeff = dndz_coeff[:, None]
        except IndexError:
            self.r0_coeff = r0_coeff
            self.r0_2 = r0_2
            self.gamma_2 = gamma_2
            self.dndz_index = dndz_index
            self.dndz_coeff = dndz_coeff

    def chi_perp_exp(self):
        a = 1 / (1 + self.z)
        norm_const = a * self.Hz / (2 * self.dv)  # 1/Mpc
        lim = self.dv / (a * self.Hz)  # Mpc

        norm_const = self.r0func * np.sqrt(np.pi) * erf(lim / self.r0func)

        antideriv = norm_const * np.exp(-((self.rho_com / self.r0func) ** 2))
        return norm_const * antideriv

    def _calc_prob(self, chi_perp):
        rate_of_incidence = (1 + chi_perp) * self.mean_dNdz()
        prob_miss = np.exp(-rate_of_incidence)
        prob_hit = 1 - prob_miss
        return prob_hit

    def phit_1halo(self):
        chi_perp1 = self.chi_perp_exp()
        prob_hit = self._calc_prob(chi_perp1 - 1)  # removing radnom term from 1-halo
        # artifically inflating the variance.
        # based on numerics
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        return prob_hit

    def phit_2halo(self):
        chi_perp2 = self.chi_perp(self.r0func_2h(), self.gamma_2)
        prob_hit = self._calc_prob(chi_perp2)
        # artifically inflating the variance.
        # based on numerics
        prob_hit = np.clip(prob_hit, 0.01, 0.99)
        return prob_hit

    def log_prior(self):
        """the Bayesian prior. Will change with each model based on which params are 
        important to the model. 

        Returns:
            ln_prior (float): natural log of the prior
        """
        r0_coeff = self.r0_coeff
        r0_2 = self.r0_2
        gamma_2 = self.gamma_2
        dndz_index = self.dndz_index
        dndz_coeff = self.dndz_coeff

        # flat prior on r0, gaussian prior on gamma around 1.6
        if (r0_coeff < 0) or (r0_coeff > 10):
            return -np.inf
        if (r0_2 < 0) or (r0_2 > 10):
            return -np.inf
        if (
            (dndz_index < -3)
            or (dndz_index > 3)
            or (dndz_coeff < 0)
            or (dndz_coeff > 40)
        ):
            return -np.inf

        sig = 1.0
        # ln_prior = -0.5 * ((gamma - 6) ** 2 / (sig) ** 2)
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

