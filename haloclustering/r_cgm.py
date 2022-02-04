import numpy as np


def compute_Rcrossing(
    model,
    flat_samples,
    thin=10,
    log=True,
    only_crossing=False,
    verbose=False,
    comoving=True,
):
    r_lin = model.rho_com
    redshift = model.z

    r21_sample = []
    count_2h_1h = 0
    for sample in flat_samples[::thin]:
        # here is where the theta is input
        model.set_params(sample)
        fc_1h = model.phit_1halo()
        fc_2h = model.phit_2halo()
        # fc_total = model.phit_sum()
        # for two crossings...R21 or R2total = crossingpoint
        try:
            crossingpoint = np.where(np.diff(np.sign(fc_2h - fc_1h)) == 2)[0][0]
        except:
            # print(list(sample), list(sample2h))
            # this means the two halo term is always greater than the 1h term?
            if np.all(np.sign(fc_2h - fc_1h)) == 1:
                if verbose:
                    print("2h > 1h")
                count_2h_1h += 1
            if np.all(np.sign(fc_2h - fc_1h)) == -1:
                if verbose:
                    print("1h > 2h")

            crossingpoint = 0

        r21 = np.mean(r_lin[crossingpoint : crossingpoint + 2])
        r21_sample.append(r21)

    # convert to arrays in logspace
    r21_sample = np.array(r21_sample, dtype=np.float32)

    # toss out zeros
    if only_crossing:
        r21_sample = np.where(r21_sample > 1e-5, r21_sample, np.nan)

    # convert to kpc
    r21_sample *= 1e3

    if not comoving:
        r21_sample /= 1 + redshift

    if log:
        r21_sample = np.log10(r21_sample)

    return r21_sample
