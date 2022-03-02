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
        params = self.sampler.get_chain(discard=discard, thin=1, flat=True)

        # instantiate model, feeding in data set
        self.model.set_params(params.T)
        fc = self.model.phit_sum()
        if bins is None:
            bins = self.bins
        else:
            bins = np.arange(0, 7, 0.1)
        r_com = self.model.rho_com

        fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic="median", bins=bins)
        fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = np.quantile(
            fc_mod_bin, [0.5, 0.16, 0.84], axis=0
        )
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
        return cf, lolim, uplim

    def plot_fc_rho(self, bins=None):
        if bins is None:
            bins = self.bins
        cf, lolim, uplim = self._compute_empirical_fc_bins(bins)
        fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = self._compute_model_fc_bins(bins)

        plt.figure(figsize=(10, 7))
        bincenters = (bins[:1] + bins[1:]) / 2
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

        plt.xlim(0, 1.5)
        plt.legend()
        plt.ylabel(r"$f_c$")
        plt.tight_layout()
        plt.xlabel(r"$R_{\perp,c}$ [Mpc]")
        plt.title(self.title)
        plt.savefig(self.savefile)
        plt.show()

