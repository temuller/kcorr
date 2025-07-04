import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

import sncosmo

class SED(object):
    """Creates a Spectral Energy Distribution (SED) object from
    an sncosmo source.
    """
    def __init__(self, 
                 source: str, 
                 z: float, 
                 mwebv: float = 0.0, 
                 phase_range: tuple[float, float] = (-10, 90), 
                 bands: list = ['ztf::g', 'ztf::r', 'ztf::i']
                 **kwargs: dict):
        """
        Parameters
        ----------
        source: sncosmo source.
        z: redshift.
        mwebv: Milky-Way dust extinction.
        """
        self.source = source
        self.z = z
        self.mwebv = mwebv
        self.load_model(source)
        self.set_parameters(**kwargs)
        # set other parameters
        self.bands = bands
        self.colours = {'ztf::g':"green", 'ztf::r':"red", 'ztf::i':"gold"}
        # time range
        self.phase_range = phase_range
        step = 0.1
        self.times = np.arange(self.phase_range[0], 
                               self.phase_range[1] + step,
                               step
                              )
    
    def load_model(self, source, mw_dust_law: sncosmo.PropagationEffect = None) -> sncosmo.models.Model:
        """Loads the SED model from an sncosmo Source.
        """
        self.model = sncosmo.Model(source=self.source)
        # Milky-Way dust law
        if mw_dust_law is None:
            mw_dust_law = sncosmo.CCM89Dust()
        self.model.add_effect(mw_dust_law, 'mw', 'obs')

    def set_parameters(self, **kwargs):
        """Sets model parameters.
        """
        self.rest_model = deepcopy(self.model)  # model @ z=0, without corrections
        self.model.set(**kwargs)
        self.model.set(z=self.z)
        self.model.set(mwebv=self.mwebv)

    def plot_lightcurves(self, zp: float = None):
        """Plots the model light curves.
        """
        if zp is None:
            zp = 30
        fig, ax = plt.subplots(figsize=(6, 4))
        for band, colour in self.colours.items():
            flux = self.model.bandflux(band, self.times, zp=zp, zpsys='ab')
            mag = -2.5 * np.log10(flux) + zp
            ax.plot(self.times, mag, label=band, color=colour)
        
        plt.gca().invert_yaxis()  # Brighter is up
        ax.set_xlabel('Days since B-maximum', fontsize=16)
        ax.set_ylabel('Apparent Magnitude', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_kcorr(self, zp: float = 30, zpsys: str = 'ab'):
        """Plots the same-filter K-correction.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        for band, colour in self.colours.items():
            rest_flux = self.rest_model.bandflux(band, self.times, zp=zp, zpsys=zpsys)
            flux = self.model.bandflux(band, self.times, zp=zp, zpsys=zpsys) 
            kcorr = -2.5 * np.log10(rest_flux / flux)
            ax.plot(self.times, kcorr, label=band, color=colour)
        
        ax.set_xlabel('Days since B-maximum', fontsize=16)
        ax.set_ylabel(r'$K$-correction (mag)', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()