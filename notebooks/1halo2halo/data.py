import glob

import cgmsquared.clustering2 as c2
from astropy.table import Table, vstack
from cgmsquared import load_cgmsquared


def get_combined_dataset(cgmsqfile=None, casbahfile=None):
    """format for returned data is (
        z,
        rho_com,
        mass,
        hits,
        misses,
        Hz,
        dv,
        cgm_data_doanly,
        do_anly,
    )

    Args:
        cgmsqfile ([type], optional): [description]. Defaults to None.
        casbahfile ([type], optional): [description]. Defaults to None.
    """

    # cgmsquared
    if cgmsqfile is None:
        surveyfile = "/Users/mwilde/python/cgm-squared/cgmsquared/data/cgm/cgmsquared_cgmsurvey_aodm_vhalos10_3sigma.json"
    else:
        surveyfile = cgmsqfile
    ion = "HI"
    cgm = load_cgmsquared.load_cgm_survey(build_sys=False, survey_file=surveyfile)
    cgm.add_ion_to_data(ion)

    # casbah
    if casbahfile is None:
        casbahfile = (
            "/Users/mwilde/Dropbox/Research/data/CASBaH/h1_galaxies_20Mpc_500kms_*.fits"
        )
    survey_files = glob.glob(casbahfile)
    cas_tab_list = []
    for cas_tab_file in survey_files:
        tab = Table.read(cas_tab_file)
        # add in useful naming conventions
        tab["rho_rvir"] = tab["rho_impact"] / tab["rvir"]
        tab["z"] = tab["z_1"]
        tab["sig_logN_HI"] = tab["sig_logN"]
        cas_tab_list.append(tab)

    cgm_data_cas = vstack(cas_tab_list)

    # format of data is
    data = c2.combine_cgm2_casbah_cluster_data(cgm, cgm_data_cas)
    return data

