import glob
import os
import pickle

import astropy.units as u
import casbah.gal_properties as caprop
import cgmsquared.clustering2 as c2
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table, hstack, vstack
from cgmsquared import load_cgmsquared


def get_combined_dataset(cgmsqfile=None, casbahfile=None, **kwargs):
    """format for returned data is (
        z,
        rho_com,
        mass,
        hits,
        misses,
        Hz,
        dv,
        rvir,
        cgm_data_doanly,
        do_anly,
    )

    Args:
        cgmsqfile (str or path object, optional): path to cgmsq data json. Defaults to None.
        casbahfile (str or path object: path to casbah folder with fits tables in it. Defaults to None.
    """

    # cgmsquared
    if cgmsqfile is None:
        surveyfile = "/Users/mwilde/python/cgm-squared/cgmsquared/data/cgm/cgmsquared_cgmsurvey_aodm_vhalos10_3sigma.json"
    else:
        surveyfile = cgmsqfile
    ion = "HI"
    cgm = load_cgmsquared.load_cgm_survey(build_sys=False, survey_file=surveyfile)
    cgm.add_ion_to_data(ion)
    cgm._data["rho_impact"] = cgm._data["rho"]

    # casbah
    if casbahfile is None:
        casbahfile = "/Users/mwilde/Dropbox/CASBaH/data/h1_galaxies_20Mpc_500kms_*.fits"
    survey_files = glob.glob(casbahfile)
    cas_tab_list = []
    for cas_tab_file in survey_files:
        tab = Table.read(cas_tab_file)
        # add in useful naming conventions
        # rho_impact in physical kpc
        tab["rho_rvir"] = tab["rho_impact"] / tab["rvir"]
        tab["rho"] = tab["rho_impact"]  # for hits_and_misses()
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
        tuple: z_lin, r_lin, m_lin, hits, misses, Hz_lin, dv, rvir, dataobj, do_anly
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
    dataobj = 1.0
    do_anly = np.ones_like(r_lin)

    data = z_lin, r_lin, m_lin, hits, misses, Hz_lin, dv, rvir, dataobj, do_anly
    return data


def get_sampler_pickle_file(pkl_file):
    if os.path.exists(pkl_file):

        infile = open(pkl_file, "rb")
        sampler = pickle.load(infile)
        infile.close()
    else:
        print(""""You need to find the output of the emcee model or rerun it""")

    return sampler


def get_fc_pickle_file(pkl_file):
    if os.path.exists(pkl_file):

        infile = open(pkl_file, "rb")
        sampler = pickle.load(infile)
        infile.close()
    else:
        print(
            """"You need to find the output of the coverfing fraction for each model or rerun it"""
        )

    return sampler


def xmatch(
    table1,
    table2,
    ra1_key,
    dec1_key,
    ra2_key,
    dec2_key,
    units="deg",
    max_sep=1.0 * u.arcsec,
):
    # convert to astropy
    if not isinstance(table1, Table):
        if isinstance(table1, pd.DataFrame):
            table1 = Table.from_pandas(table1)
        else:
            print("table1 must be pandas or astropy table")

    if not isinstance(table2, Table):
        if isinstance(table2, pd.DataFrame):
            table2 = Table.from_pandas(table2)
        else:
            print("table2 must be pandas or astropy table")

    ra1 = np.array(table1[ra1_key])
    dec1 = np.array(table1[dec1_key])
    ra2 = np.array(table2[ra2_key])
    dec2 = np.array(table2[dec2_key])

    c1 = SkyCoord(ra=ra1, dec=dec1, unit=units)
    c2 = SkyCoord(ra=ra2, dec=dec2, unit=units)

    # find the closest match
    idx, d2d, _ = c1.match_to_catalog_sky(c2, nthneighbor=1)

    sep_constraint = d2d < max_sep
    t1_matches = table1[sep_constraint]
    t2_matches = table2[idx[sep_constraint]]

    comb_tab = hstack([t1_matches, t2_matches])

    # add ang_sep
    comb_tab["ang_sep"] = d2d[sep_constraint].arcsec
    return comb_tab
