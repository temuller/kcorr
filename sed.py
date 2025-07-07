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
                 bands: list = ['ztf::g', 'ztf::r', 'ztf::i'],
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
        # load model and set parameters
        self.load_model(source)
        params_dict = {"z":z, "mwebv":mwebv} | kwargs
        self.set_parameters(**params_dict)
        # set bands and plot params
        self.bands = bands
        self._set_wavelength_coverage()
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
    
    def _set_wavelength_coverage(self):
        bands_wave = np.empty(0)
        for band in self.bands:
            bands_wave = np.r_[bands_wave, sncosmo.get_bandpass(band).wave]
        self.minwave = bands_wave.min()
        self.maxwave = bands_wave.max()

    def plot_lightcurves(self, zp: float = 30, zpsys = 'ab', restframe: bool = False):
        """Plots the model light curves.
        """
        # chose between observer- and rest-frame model
        if restframe is True:
            model = self.rest_model
        else:
            model = self.model
        # plot light curves
        fig, ax = plt.subplots(figsize=(6, 4))
        for band, colour in self.colours.items():
            flux = model.bandflux(band, self.times, zp=zp, zpsys=zpsys)
            mag = -2.5 * np.log10(flux) + zp
            ax.plot(self.times, mag, label=band, color=colour)
        # config
        plt.gca().invert_yaxis()
        ax.set_xlabel('Days since B-maximum', fontsize=16)
        ax.set_ylabel('Apparent Magnitude', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_sed(self, phase: float = 0.0, minwave: float = None, maxwave: float = None):
        """Plots the SED model at a given phase.
        """
        phase = np.array(phase)
        if minwave is None:
            minwave = self.minwave
        if maxwave is None:
            maxwave = self.maxwave
        # get flux
        rest_wave = np.arange(self.rest_model.minwave(), self.rest_model.maxwave() )
        rest_flux = self.rest_model.flux(phase, rest_wave)
        wave = np.arange(self.model.minwave(), self.model.maxwave())
        flux = self.model.flux(phase, wave)
        # plot SED
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rest_wave, rest_flux, label="Rest-frame")
        ax.plot(wave, flux, label="Observer-frame")
        # plot filters
        ax2 = ax.twinx() 
        for band in self.bands:
            band_wave = sncosmo.get_bandpass(band).wave
            band_trans = sncosmo.get_bandpass(band).trans
            ax2.plot(band_wave, band_trans, color=self.colours[band], alpha=0.4)
        # config
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
        ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
        ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
        ax.tick_params('both', labelsize=14)
        ax.set_xlim(minwave, maxwave)
        ax2.set_ylim(None, 8)
        ax2.set_yticks([])
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