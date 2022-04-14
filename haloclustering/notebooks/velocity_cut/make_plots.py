import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import haloclustering.data as datamodule
import pandas as pd

plt.rcParams["font.serif"] = "DejaVu Serif"
plt.rcParams["font.family"] = "serif"

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 16

# get data
df = pd.read_csv(
    "/Users/mwilde/python/haloclustering/haloclustering/data/combined_df.csv"
)
df["log_rho"] = np.log10(df.rho)
df.sort_values("rho", inplace=True)

df.dropna(subset=["mstars"], inplace=True)

