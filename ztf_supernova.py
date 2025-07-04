import sys
sys.path.insert(0, '../ztf_early_bumps/')
from supernova import Supernova

import pandas as pd

import ztfidr
ztf_sample = ztfidr.get_sample()
ztf_df = ztf_sample.get_data()

def load_sn(ztfname: str) -> [pd.DataFrame, float]:
    """Loads a ZTF supernova and its properties.
    """
    # load SN and info
    sn = Supernova(ztfname, tmax=100)
    sn_info = ztf_sample.data.loc[ztfname]
    sn_df = sn.lc_df.copy()
    # rename filters
    for band in sn_df["filter"].unique():
        model_band = f"{band[:-1]}::{band[-1]}"  # e.g. ztf::g
        sn_df.replace(band, model_band, inplace=True)
    # get properties
    z = sn_info.redshift
    t0 = sn_info.t0
    mwebv = sn_info.mwebv
    # get phases
    sn_df["phase"] = (sn.lc_df.mjd.values - t0) / (1 + z)
    # rename some columns and add mag system
    sn_df.rename(columns={"filter": "band", "ZP":"zp"}, inplace=True)
    sn_df["zpsys"] = "ab"
    columns = ["mjd", "phase", "band", "flux", "flux_err", "zp", "zpsys"]
    sn_df = sn_df[columns]

    return sn_df, z, mwebv, t0