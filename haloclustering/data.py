import glob

import cgmsquared.clustering2 as c2
from astropy.table import Table, vstack
from cgmsquared import load_cgmsquared
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import casbah.gal_properties as caprop


def get_combined_dataset(cgmsqfile=None, casbahfile=None, **kwargs):
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
        casbahfile = "/Users/mwilde/Dropbox/CASBaH/data/h1_galaxies_20Mpc_500kms_*.fits"
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
    data = c2.combine_cgm2_casbah_cluster_data(cgm, cgm_data_cas, **kwargs)
    return data


def make_grid_data(mass, redshift):
    """z_lin, r_lin, m_lin, hits, misses, Hz_lin, dv, rvir, do_anly

    Args:
        mass (float): generate data at single mass
        redshift (float): generate data at sinlge z

    Returns:
        tuple: z_lin, r_lin, m_lin, hits, misses, Hz_lin, dv, rvir, do_anly
    """
    npts = 1000
    r_lin = np.geomspace(1e-10, 20, npts)
    m_lin = np.full(npts, 10 ** mass)
    z_lin = np.full(npts, redshift)
    Hz_lin = cosmo.H(z_lin).value
    dv = 500.0
    hits = np.ones_like(r_lin)
    misses = np.ones_like(r_lin)
    logmhalo = caprop.calchalomass(mass, redshift)
    rvir = caprop.calcrvir(logmhalo, redshift)
    do_anly = np.ones_like(r_lin)

    data = z_lin, r_lin, m_lin, hits, misses, Hz_lin, dv, rvir, do_anly
    return data
