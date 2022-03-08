import numpy as np
import matplotlib.pyplot as plt

# from haloclustering.data import get_combined_dataset
from scipy.stats import binned_statistic
from haloclustering import models
from haloclustering import data as datamodule
from casbah import cgm
import matplotlib

matplotlib.rcParams["font.serif"] = "DejaVu Serif"
matplotlib.rcParams["font.family"] = "serif"

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 16


class CovFracPlot:
    def __init__(self, data, model, sampler) -> None:
        self.data = data
        self.model = model
        self.sampler = sampler
        self.discard = 1000
        self.title = None
        self.label = None
        self.savefile = None
        self.r_com = self.data[1]
        self.bins = np.arange(0, 7, 0.1)

    def _compute_median_model_fc_bins(self, bins=None):
        # get median fit params
        flat_samples = self.sampler.get_chain(discard=self.discard, thin=1, flat=True)
        params = np.median(flat_samples, axis=0)

        # instantiate model, feeding in data set
        self.model.set_params(params)
        fc = self.model.phit_sum()

        r_com = self.r_com
        if bins is None:
            bins = self.bins
        fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic="median", bins=bins)
        fc_mod_bin_low, _, _ = binned_statistic(
            r_com, fc, statistic=lambda x: np.quantile(x, 0.16), bins=bins
        )
        fc_mod_bin_high, _, _ = binned_statistic(
            r_com, fc, statistic=lambda x: np.quantile(x, 0.84), bins=bins
        )
        return fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high

    def _compute_model_fc_bins(self, bins=None):
        """Compute the covering fraction over all samples space by marginalizing over 
        galaxy space. Then return the median, lower and upper 1-sigma quantiles of the 
        covering fraction per impact parameter bin. 

        Args:
            bins (int, optional): _description_. Defaults to None.

        Returns:
            tuple : fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high
        """
        # model data covering fraction
        discard = 1000
        params = self.sampler.get_chain(discard=discard, thin=10, flat=True)

        # instantiate model, feeding in data set
        self.model.set_params(params.T)
        fc = self.model.phit_sum()
        if bins is None:
            bins = self.bins
        r_com = self.model.rho_com

        fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic="mean", bins=bins)
        fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = np.quantile(
            fc_mod_bin, [0.5, 0.16, 0.84], axis=0
        )
        self.fc_mod_bin = fc_mod_bin
        self.fc_mod_bin_low = fc_mod_bin_low
        self.fc_mod_bin_high = fc_mod_bin_high
        return fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high

    def _compute_empirical_fc_bins(self, bins=None):
        # empirical data covering fraction
        dataobj = self.data[-2]
        ion = "HI"
        thresh = 14.0
        attr = "rho_com"
        if bins is None:
            bins = self.bins
        cf, lolim, uplim = cgm.cov_frac(
            dataobj, ion, thresh=thresh, attr=attr, bins=bins
        )
        self.cf = cf
        self.lolim = lolim
        self.uplim = uplim
        return cf, lolim, uplim

    def plot_fc_rho(self, bins=None):
        if bins is None:
            bins = self.bins
        cf = self.cf
        lolim = self.lolim
        uplim = self.uplim
        # cf, lolim, uplim = self._compute_empirical_fc_bins(bins)
        # fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = self._compute_model_fc_bins(bins)
        fc_mod_bin = self.fc_mod_bin
        fc_mod_bin_low = self.fc_mod_bin_low
        fc_mod_bin_high = self.fc_mod_bin_high

        plt.figure(figsize=(10, 7))
        bincenters = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(
            bincenters,
            cf,
            yerr=[lolim, uplim],
            marker=".",
            color="black",  # fmt='.k',
            ecolor="lightgrey",
            elinewidth=4,
            capsize=0,
            label="data",
        )

        plt.errorbar(
            bincenters,
            fc_mod_bin,
            yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
            marker=".",
            color="tab:blue",
            ecolor="lightblue",
            elinewidth=4,
            capsize=0,
            label=self.label,
        )

        plt.xlim(bins.min(), bins.max())
        plt.legend()
        plt.ylabel(r"$f_c$")
        plt.tight_layout()
        plt.xlabel(r"$R_{\perp,c}$ [Mpc]")
        plt.title(self.title)
        plt.savefig(self.savefile)
        plt.show()


class MultiCovFracPlot(CovFracPlot):
    def __init__(self, data, model, sampler, model2, sampler2) -> None:
        super().__init__(data, model, sampler)
        self.model2 = model2
        self.sampler2 = sampler2
        self.label2 = None

    def plot_fc_rho(self, bins=None):
        if bins is None:
            bins = self.bins
        cf, lolim, uplim = self._compute_empirical_fc_bins(bins)
        fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = self._compute_model_fc_bins(
            self.model, self.sampler, bins
        )
        fc_mod_bin2, fc_mod_bin_low2, fc_mod_bin_high2 = self._compute_model_fc_bins(
            self.model2, self.sampler2, bins
        )

        # plt.figure(figsize=(10, 7))
        bincenters = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(
            bincenters,
            cf,
            yerr=[lolim, uplim],
            marker=".",
            color="black",  # fmt='.k',
            ecolor="lightgrey",
            elinewidth=4,
            capsize=0,
            label="data",
        )

        plt.errorbar(
            bincenters,
            fc_mod_bin,
            yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
            marker=".",
            color="tab:blue",
            ecolor="lightblue",
            elinewidth=4,
            capsize=0,
            label=self.label,
        )

        plt.errorbar(
            bincenters,
            fc_mod_bin2,
            yerr=[fc_mod_bin2 - fc_mod_bin_low2, fc_mod_bin_high2 - fc_mod_bin2],
            marker=".",
            color="tab:green",
            ecolor="lightgreen",
            elinewidth=4,
            capsize=0,
            label=self.label,
        )
        plt.xlim(bins.min(), bins.max())
        plt.legend()
        plt.ylabel(r"$f_c$")
        plt.tight_layout()
        plt.xlabel(r"$R_{\perp,c}$ [Mpc]")
        plt.title(self.title)
        plt.savefig(self.savefile)
        plt.show()

    def _compute_model_fc_bins(self, model, sampler, bins=None):
        """Compute the covering fraction over all samples space by marginalizing over 
        galaxy space. Then return the median, lower and upper 1-sigma quantiles of the 
        covering fraction per impact parameter bin. 

        Args:
            bins (int, optional): _description_. Defaults to None.

        Returns:
            tuple : fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high
        """
        # model data covering fraction
        discard = 1000
        params = sampler.get_chain(discard=discard, thin=10, flat=True)

        # instantiate model, feeding in data set
        model.set_params(params.T)
        fc = model.phit_sum()
        if bins is None:
            bins = self.bins
        r_com = model.rho_com

        fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic="mean", bins=bins)
        fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = np.quantile(
            fc_mod_bin, [0.5, 0.16, 0.84], axis=0
        )
        return fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high


def compute_model_fc(model, sampler):
    # instantiate model, feeding in data set
    discard = 1000
    params = sampler.get_chain(discard=discard, thin=10, flat=True)
    model.set_params(params.T)
    fc = model.phit_sum()
    r_com = model.rho_com
    return r_com, fc


def compute_model_fc_bins(r_com, fc, bins):
    """Compute the covering fraction over all samples space by marginalizing over 
    galaxy space. Then return the median, lower and upper 1-sigma quantiles of the 
    covering fraction per impact parameter bin. 

    Args:
        bins (int, optional): _description_. Defaults to None.

    Returns:
        tuple : fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high
    """
    fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic="mean", bins=bins)
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = np.quantile(
        fc_mod_bin, [0.5, 0.16, 0.84], axis=0
    )
    return fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high


def compute_empirical_fc_bins(data, bins=None):
    # empirical data covering fraction
    dataobj = data[-2]
    ion = "HI"
    thresh = 14.0
    attr = "rho_com"
    cf, lolim, uplim = cgm.cov_frac(dataobj, ion, thresh=thresh, attr=attr, bins=bins)
    return cf, lolim, uplim


def plot_fc(data, bins, savefile, masslabel):
    model = models.Model2h(data=data)
    model2 = models.Model(data=data)
    model3 = models.rvirModel(data=data)
    model4 = models.Model1hBeta(data=data)
    base_dir = "/Users/mwilde/python/haloclustering/haloclustering/notebooks/"
    model_2h_sampler_file = base_dir + "two-halo-only/model_2h_only_sampler.pkl"
    model_1beta_sampler_file = (
        base_dir + "base_model/model_1beta_mass_dependence_sampler.pkl"
    )
    model_rvir_sampler_file = base_dir + "rvir_as_r0/model_rvir_as_r0_sampler.pkl"
    model_1hbeta_sampler_file = (
        base_dir + "1halo_with_beta/model_1h_with_beta_sampler.pkl"
    )
    sampler = datamodule.get_sampler_pickle_file(model_2h_sampler_file)
    sampler2 = datamodule.get_sampler_pickle_file(model_1beta_sampler_file)
    sampler3 = datamodule.get_sampler_pickle_file(model_rvir_sampler_file)
    sampler4 = datamodule.get_sampler_pickle_file(model_1hbeta_sampler_file)

    # precompute fc
    r_com1, fc1 = compute_model_fc(model, sampler)
    r_com2, fc2 = compute_model_fc(model2, sampler2)
    r_com3, fc3 = compute_model_fc(model3, sampler3)
    r_com4, fc4 = compute_model_fc(model4, sampler4)

    # bin up the covering fraction data in rho_impact
    cf, lolim, uplim = compute_empirical_fc_bins(data, bins)
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = compute_model_fc_bins(
        r_com1, fc1, bins
    )
    fc_mod_bin2, fc_mod_bin_low2, fc_mod_bin_high2 = compute_model_fc_bins(
        r_com2, fc2, bins
    )
    fc_mod_bin3, fc_mod_bin_low3, fc_mod_bin_high3 = compute_model_fc_bins(
        r_com3, fc3, bins
    )
    fc_mod_bin4, fc_mod_bin_low4, fc_mod_bin_high4 = compute_model_fc_bins(
        r_com4, fc4, bins
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    bincenters = (bins[:-1] + bins[1:]) / 2

    kwargs = {"elinewidth": 2, "capsize": 5, "lw": 2, "alpha": 1}
    ax.errorbar(
        bincenters,
        cf,
        yerr=[lolim, uplim],
        marker=".",
        color="black",  # fmt='.k',
        ecolor="lightgrey",
        label="data",
        **kwargs
    )

    ax.errorbar(
        bincenters,
        fc_mod_bin,
        yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
        marker=".",
        color="tab:blue",
        ecolor="lightblue",
        label=r"2$^h$ only model",
        **kwargs
    )

    ax.errorbar(
        bincenters,
        fc_mod_bin2,
        yerr=[fc_mod_bin2 - fc_mod_bin_low2, fc_mod_bin_high2 - fc_mod_bin2],
        marker=".",
        color="tab:purple",
        ecolor="tab:purple",
        label=r"1$^h$ + 2$^h$",
        **kwargs
    )
    ax.errorbar(
        bincenters,
        fc_mod_bin3,
        yerr=[fc_mod_bin3 - fc_mod_bin_low3, fc_mod_bin_high3 - fc_mod_bin3],
        marker=".",
        color="tab:pink",
        ecolor="tab:pink",
        label=r"$r_0 \sim R_{vir}$",
        **kwargs
    )
    ax.errorbar(
        bincenters,
        fc_mod_bin4,
        yerr=[fc_mod_bin4 - fc_mod_bin_low4, fc_mod_bin_high4 - fc_mod_bin4],
        marker=".",
        color="tab:green",
        ecolor="tab:green",
        label=r"1$^h$ w/beta",
        **kwargs
    )
    ax.set_xlim(bins.min(), bins.max())
    ax.set_ylim(0, 1)
    ax.text(3, 0.5, masslabel)
    plt.legend()
    plt.ylabel(r"$f_c$")
    plt.xscale("log")
    plt.tight_layout()
    plt.xlabel(r"$R_{\perp,c}$ [Mpc]")
    plt.savefig(savefile)
