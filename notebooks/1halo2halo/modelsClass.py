import cgmsquared.clustering2 as c2
from cgmsquared import clustering as cgm2_cluster
from numpy.core.fromnumeric import mean


class Model:
    def __init__(self, data, theta) -> None:
        self.data = None
        self.theta = None
        # self.m0 = None

        # z, rho_com, mass, hits, misses, Hz, dv, cgm_data_doanly, do_anly = self.data

    def chi_perp(self, theta, data):
        m0, r0, gamma, beta, dndz_index, dndz_coeff = theta
        z, rho_com, mass, hits, misses, Hz, dv, cgm_data_doanly, do_anly = data

        r0_mass = r0 * (mass / m0) ** (beta)
        chi_i = c2.chi_perp_analytic(r0_mass, gamma, rho_com, z, Hz, dv)
        return chi_i

    def mean_dNdz(self, theta, data, vmax=500.0):
        # dN_dz for HI with logNHI > 14
        m0, r0, gamma, beta, dndz_index, dndz_coeff = theta
        z, rho_com, mass, hits, misses, Hz, dv, cgm_data_doanly, do_anly = data
        ion_lz = cgm2_cluster.hi_lz(
            z, danforth=False, gamma=dndz_index, coeff=dndz_coeff
        )
        clight = 299792.458
        # mean number of absorbers along line of sight in dz window
        dz = 2 * (1 + z) * (vmax / clight)
        mean_dN_dz = ion_lz * dz  # ~ (1+z)^3.3
        return mean_dN_dz

    def phit_halo(self, theta, data):
        pass

    def phit_sum(self, chi_1h, chi_2h):
        pass

