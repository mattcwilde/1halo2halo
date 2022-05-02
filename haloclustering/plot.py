import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    fc_mod_bin, _, _ = binned_statistic(r_com, fc, statistic=np.nanmean, bins=bins)
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = np.quantile(
        fc_mod_bin, [0.5, 0.16, 0.84], axis=0
    )
    return fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high


def compute_empirical_fc_bins(data, bins=None, mass_cut=None):
    # empirical data covering fraction
    if mass_cut is not None:
        dataobj = data[-2][mass_cut]
    else:
        dataobj = data[-2]
    ion = "HI"
    thresh = 14.0
    attr = "rho_com"
    cf, lolim, uplim = cgm.cov_frac(dataobj, ion, thresh=thresh, attr=attr, bins=bins)
    return cf, lolim, uplim


def plot_fc(ax, data, bins, masslabel, rerun_fc=False, mass_cut=None):
    model = models.Model2h(data=data)
    model2 = models.Model(data=data)
    model3 = models.rvirModel(data=data)
    model4 = models.Model2hBeta(data=data)
    base_dir = "/Users/mwilde/python/haloclustering/haloclustering/notebooks/"
    model_2h_sampler_file = base_dir + "two-halo-only/model_2h_only_sampler.pkl"
    model_1beta_sampler_file = (
        base_dir + "base_model/model_1beta_mass_dependence_sampler.pkl"
    )
    model_rvir_sampler_file = base_dir + "rvir_as_r0/model_rvir_as_r0_sampler.pkl"
    model_2hbeta_sampler_file = (
        base_dir + "two-halo-only-w-beta/model_2h_only_with_beta_sampler.pkl"
    )
    sampler = datamodule.get_sampler_pickle_file(model_2h_sampler_file)
    sampler2 = datamodule.get_sampler_pickle_file(model_1beta_sampler_file)
    sampler3 = datamodule.get_sampler_pickle_file(model_rvir_sampler_file)
    sampler4 = datamodule.get_sampler_pickle_file(model_2hbeta_sampler_file)

    if rerun_fc:
        import pickle

        # precompute fc
        r_com1, fc1 = compute_model_fc(model, sampler)
        r_com2, fc2 = compute_model_fc(model2, sampler2)
        r_com3, fc3 = compute_model_fc(model3, sampler3)
        r_com4, fc4 = compute_model_fc(model4, sampler4)

        # save the sampler
        with open("model_2h_only_fc.pkl", "wb") as f:
            pickle.dump(fc1, f)
        with open("model_1beta_mass_dependence_fc.pkl", "wb") as f:
            pickle.dump(fc2, f)
        with open("model_rvir_as_r0_fc.pkl", "wb") as f:
            pickle.dump(fc3, f)
        with open("model_2h_only_with_beta_fc.pkl", "wb") as f:
            pickle.dump(fc4, f)
    else:
        fc1 = datamodule.get_fc_pickle_file("../../data/model_2h_only_fc.pkl")
        fc2 = datamodule.get_fc_pickle_file(
            "../../data/model_1beta_mass_dependence_fc.pkl"
        )
        fc3 = datamodule.get_fc_pickle_file("../../data/model_rvir_as_r0_fc.pkl")
        fc4 = datamodule.get_fc_pickle_file("../../data/model_2h_only_with_beta_fc.pkl")

    r_com = model.rho_com[mass_cut]

    # bin up the covering fraction data in rho_impact
    cf, lolim, uplim = compute_empirical_fc_bins(data, bins, mass_cut)
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = compute_model_fc_bins(
        r_com, fc1[:, mass_cut], bins
    )
    fc_mod_bin2, fc_mod_bin_low2, fc_mod_bin_high2 = compute_model_fc_bins(
        r_com, fc2[:, mass_cut], bins
    )
    fc_mod_bin3, fc_mod_bin_low3, fc_mod_bin_high3 = compute_model_fc_bins(
        r_com, fc3[:, mass_cut], bins
    )
    fc_mod_bin4, fc_mod_bin_low4, fc_mod_bin_high4 = compute_model_fc_bins(
        r_com, fc4[:, mass_cut], bins
    )

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
        **kwargs,
    )

    ax.errorbar(
        bincenters,
        fc_mod_bin,
        yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
        marker=".",
        color="tab:blue",
        ecolor="lightblue",
        label=r"2$^h$ only",
        **kwargs,
    )
    ax.errorbar(
        bincenters,
        fc_mod_bin4,
        yerr=[fc_mod_bin4 - fc_mod_bin_low4, fc_mod_bin_high4 - fc_mod_bin4],
        marker=".",
        color="tab:green",
        ecolor="tab:green",
        label=r"2$^h$ w/ $\beta$",
        **kwargs,
    )

    ax.errorbar(
        bincenters,
        fc_mod_bin2,
        yerr=[fc_mod_bin2 - fc_mod_bin_low2, fc_mod_bin_high2 - fc_mod_bin2],
        marker=".",
        color="tab:purple",
        ecolor="tab:purple",
        label=r"1$^h$ + 2$^h$",
        **kwargs,
    )
    ax.errorbar(
        bincenters,
        fc_mod_bin3,
        yerr=[fc_mod_bin3 - fc_mod_bin_low3, fc_mod_bin_high3 - fc_mod_bin3],
        marker=".",
        color="tab:pink",
        ecolor="tab:pink",
        label=r"$r_0 \sim R_{vir}$",
        **kwargs,
    )

    ax.set_xlim(bins.min(), bins.max())
    ax.set_ylim(0, 1)
    ax.text(0.3, 0.8, masslabel, transform=ax.transAxes)
    ax.set_ylabel(r"$f_c$")
    ax.set_xscale("log")
    # plt.tight_layout()
    # ax.set_xlabel(r"$R_{\perp,c}$ [Mpc]")

    return ax


def plot_fc_diff(ax, data, bins, masslabel, rerun_fc=False, mass_cut=None):
    model = models.Model2h(data=data)
    model2 = models.Model(data=data)
    model3 = models.rvirModel(data=data)
    model4 = models.Model2hBeta(data=data)
    base_dir = "/Users/mwilde/python/haloclustering/haloclustering/notebooks/"
    model_2h_sampler_file = base_dir + "two-halo-only/model_2h_only_sampler.pkl"
    model_1beta_sampler_file = (
        base_dir + "base_model/model_1beta_mass_dependence_sampler.pkl"
    )
    model_rvir_sampler_file = base_dir + "rvir_as_r0/model_rvir_as_r0_sampler.pkl"
    model_2hbeta_sampler_file = (
        base_dir + "two-halo-only-w-beta/model_2h_only_with_beta_sampler.pkl"
    )
    sampler = datamodule.get_sampler_pickle_file(model_2h_sampler_file)
    sampler2 = datamodule.get_sampler_pickle_file(model_1beta_sampler_file)
    sampler3 = datamodule.get_sampler_pickle_file(model_rvir_sampler_file)
    sampler4 = datamodule.get_sampler_pickle_file(model_2hbeta_sampler_file)

    if rerun_fc:
        import pickle

        # precompute fc
        r_com1, fc1 = compute_model_fc(model, sampler)
        r_com2, fc2 = compute_model_fc(model2, sampler2)
        r_com3, fc3 = compute_model_fc(model3, sampler3)
        r_com4, fc4 = compute_model_fc(model4, sampler4)

        # save the sampler
        with open("model_2h_only_fc.pkl", "wb") as f:
            pickle.dump(fc1, f)
        with open("model_1beta_mass_dependence_fc.pkl", "wb") as f:
            pickle.dump(fc2, f)
        with open("model_rvir_as_r0_fc.pkl", "wb") as f:
            pickle.dump(fc3, f)
        with open("model_2h_only_with_beta_fc.pkl", "wb") as f:
            pickle.dump(fc4, f)
    else:
        fc1 = datamodule.get_fc_pickle_file("../../data/model_2h_only_fc.pkl")
        fc2 = datamodule.get_fc_pickle_file(
            "../../data/model_1beta_mass_dependence_fc.pkl"
        )
        fc3 = datamodule.get_fc_pickle_file("../../data/model_rvir_as_r0_fc.pkl")
        fc4 = datamodule.get_fc_pickle_file("../../data/model_2h_only_with_beta_fc.pkl")

    r_com = model.rho_com[mass_cut]

    # bin up the covering fraction data in rho_impact
    cf, lolim, uplim = compute_empirical_fc_bins(data, bins, mass_cut)
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = compute_model_fc_bins(
        r_com, fc1[:, mass_cut], bins
    )
    fc_mod_bin2, fc_mod_bin_low2, fc_mod_bin_high2 = compute_model_fc_bins(
        r_com, fc2[:, mass_cut], bins
    )
    fc_mod_bin3, fc_mod_bin_low3, fc_mod_bin_high3 = compute_model_fc_bins(
        r_com, fc3[:, mass_cut], bins
    )
    fc_mod_bin4, fc_mod_bin_low4, fc_mod_bin_high4 = compute_model_fc_bins(
        r_com, fc4[:, mass_cut], bins
    )

    bincenters = (bins[:-1] + bins[1:]) / 2

    kwargs = {"elinewidth": 2, "capsize": 5, "lw": 2, "alpha": 1}

    ax.errorbar(
        bincenters,
        cf - fc_mod_bin,
        yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
        marker=".",
        color="tab:blue",
        ecolor="lightblue",
        label=r"2$^h$ only model",
        **kwargs,
    )

    ax.errorbar(
        bincenters,
        cf - fc_mod_bin2,
        yerr=[fc_mod_bin2 - fc_mod_bin_low2, fc_mod_bin_high2 - fc_mod_bin2],
        marker=".",
        color="tab:purple",
        ecolor="tab:purple",
        label=r"1$^h$ + 2$^h$",
        **kwargs,
    )
    ax.errorbar(
        bincenters,
        cf - fc_mod_bin3,
        yerr=[fc_mod_bin3 - fc_mod_bin_low3, fc_mod_bin_high3 - fc_mod_bin3],
        marker=".",
        color="tab:pink",
        ecolor="tab:pink",
        label=r"$r_0 \sim R_{vir}$",
        **kwargs,
    )
    ax.errorbar(
        bincenters,
        cf - fc_mod_bin4,
        yerr=[fc_mod_bin4 - fc_mod_bin_low4, fc_mod_bin_high4 - fc_mod_bin4],
        marker=".",
        color="tab:green",
        ecolor="tab:green",
        label=r"1$^h$ w/beta",
        **kwargs,
    )
    ax.set_xlim(bins.min(), bins.max())
    ax.set_ylim(-0.3, 0.3)
    ax.text(0.3, 0.9, masslabel, transform=ax.transAxes)
    ax.set_ylabel(r"$data - f_c$")
    ax.set_xscale("log")
    ax.fill_between(bincenters, -lolim, uplim, color="k", alpha=0.1)
    ax.axhline(0, c="k", ls="--")
    # plt.tight_layout()
    # ax.set_xlabel(r"$R_{\perp,c}$ [Mpc]")

    return ax


def model_v_emp_plot(ax, df, bins, cf_list, masslabel=None):
    """Plot the model vs empirical data

    Args:
        ax (ax): matplotlib axis
        df (pandas.DataFrame): pandas df with the data
        bins (array): radial bins
        cf_list (_type_): list of covering fraction data
        masslabel (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    rho_com = df.rho.values

    colors = ["tab:blue", "tab:purple", "tab:pink"]
    labels = ["0-100 km/s", "200-300 km/s", "400-500 km/s"]
    cf_types = ["HM_0_100", "HM_200_300", "HM_400_500"]
    for cf, cf_type, color, label in zip(cf_list.T, cf_types, colors, labels):

        active_df = df.query(f'{cf_type} != "inconclusive"').copy()
        active_df["response"] = active_df[cf_type] == "hit"

        sns.regplot(
            data=active_df,
            x="rho",
            y="response",
            fit_reg=False,
            x_bins=bins,
            x_ci=68,
            ci=68,
            order=3,
            ax=ax,
            color=color,
            label=label,
        )

        ax.plot(rho_com, cf, c=color, ls="--", label=None)
        ax.set_ylabel(r"$f_c$")
    if masslabel is not None:
        ax.text(0.3, 0.9, masslabel, transform=ax.transAxes)

    # ax.set_yscale("log")
    ax.set_xscale("log")

    return ax


def new_plot_fc(ax, data, bins, masslabel, df, flat_phit_samps, mass_cut=None):

    # deal with the single power law w/beta
    model4 = models.Model2hBeta(data=data)
    fc4 = datamodule.get_fc_pickle_file("../../data/model_2h_only_with_beta_fc.pkl")
    r_com4 = model4.rho_com[mass_cut]

    # bin up the covering fraction data in rho_impact
    cf, lolim, uplim = compute_empirical_fc_bins(data, bins, mass_cut)
    fc_mod_bin4, fc_mod_bin_low4, fc_mod_bin_high4 = compute_model_fc_bins(
        r_com4, fc4[:, mass_cut], bins
    )

    # deal with the new model
    rho = df.rho.values
    fc_mod_bin, fc_mod_bin_low, fc_mod_bin_high = compute_model_fc_bins(
        rho, flat_phit_samps, bins
    )

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
        **kwargs,
    )

    ax.errorbar(
        bincenters,
        fc_mod_bin,
        yerr=[fc_mod_bin - fc_mod_bin_low, fc_mod_bin_high - fc_mod_bin],
        marker=".",
        color="tab:blue",
        ecolor="lightblue",
        label=r"2$^h$ only",
        **kwargs,
    )
    ax.errorbar(
        bincenters,
        fc_mod_bin4,
        yerr=[fc_mod_bin4 - fc_mod_bin_low4, fc_mod_bin_high4 - fc_mod_bin4],
        marker=".",
        color="tab:green",
        ecolor="tab:green",
        label=r"2$^h$ w/ $\beta$",
        **kwargs,
    )

    ax.set_xlim(bins.min(), bins.max())
    ax.set_ylim(0, 1)
    ax.text(0.3, 0.8, masslabel, transform=ax.transAxes)
    ax.set_ylabel(r"$f_c$")
    ax.set_xscale("log")
    # plt.tight_layout()
    # ax.set_xlabel(r"$R_{\perp,c}$ [Mpc]")

    return ax
