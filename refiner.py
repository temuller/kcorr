import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import partial

import sncosmo
from astropy.table import Table
from photometry import Photometry
from sed import SED
from gp_fit import fit_gp_model, gp_predict

class Refiner(object):
    """Object for building a colour-matched SED model.
    """
    def __init__(self, 
                 data: str | pd.DataFrame | Table,
                 source: str,
                 z: float,
                 mwebv: float = 0.0, 
                 phase_range: tuple[float, float] = (-10, 90),
                ):
        """
        Parameters
        ----------
        data: supernova photometry in sncosmo format.
        source: sncosmo source.
        z: redshift.
        mwebv: Milky-Way dust extinction.
        phase_range: phase range to be used for the SED model. 
        """
        self.source = source
        self.z = z
        self.mwebv = mwebv
        self.phase_range = phase_range
        self.phot = Photometry(data)
        self.bands = np.unique(self.phot.band)
        self.sed = SED(source, z, mwebv, phase_range, np.unique(self.bands))
        
    def _setup_phase(self, t0: float):
        """Loads the photometry of a supernova.
        """
        self.phot.phase = (self.phot.time - t0) / (1 + self.sed.z)

    def match_sed(self, t0: float, k1: str = 'ExpSquared', fit_mean: bool = True):
        """Modifies the SED model to match the observations using Gaussian Process (GP) regression.

        Parameters
        ----------
        t0: time of reference (e.g. optical peak).
        k1: GP kernel for the time axis.
        fit_mean: whether to fit a mean function (constant).
        """
        # select phase range
        self._setup_phase(t0)
        minphase, maxphase = self.phase_range
        self.phase_mask = (minphase <= self.phot.phase) & (self.phot.phase <= maxphase)
        # flux ratios
        model_flux = self.sed.model.bandflux(self.phot.band, 
                                             self.phot.phase,
                                             zp=self.phot.zp, 
                                             zpsys=self.phot.zpsys)
        self.ratio_flux = self.phot.flux / model_flux
        self.ratio_error = self.phot.flux_err / model_flux
        # fit mangling surface
        self.gp_model = fit_gp_model(self.phot.phase[self.phase_mask], 
                                     self.phot.eff_wave[self.phase_mask], 
                                     self.ratio_flux[self.phase_mask], 
                                     self.ratio_error[self.phase_mask], 
                                     k1=k1, fit_mean=fit_mean)
        self.gp_predict = partial(gp_predict, 
                                  ratio_pred=self.ratio_flux[self.phase_mask],
                                  error_pred=self.ratio_error[self.phase_mask],
                                  gp_model=self.gp_model,
                                 )

    def plot_fit(self):
        """Plots the light-curve fit and ratio between the observations and SED.
        """
        # phase range to use
        minphase, maxphase = self.phase_range
        pred_phase = np.arange(minphase, maxphase + 0.1, 0.1)
        
        fig, ax = plt.subplots(2, 1, height_ratios=(3, 1), gridspec_kw={"hspace":0})
        for band in self.bands:
            band_mask = self.phot.band == band
            mask = self.phase_mask & band_mask
            # apply mask
            phase = self.phot.phase[mask]
            flux, flux_err = self.phot.flux[mask], self.phot.flux_err[mask]
            zp, zpsys = self.phot.zp[mask], self.phot.zpsys[mask]
            eff_wave = sncosmo.get_bandpass(band).wave_eff
            # ratios
            ratio_flux = self.ratio_flux[mask]
            ratio_error = self.ratio_error[mask]
            
            ########################
            # observer-frame model #
            ########################            
            pred_obs_wave = np.array([eff_wave] * len(pred_phase))
            obs_ratio_fit, obs_var_fit = self.gp_predict(pred_phase, pred_obs_wave)
            obs_std_fit = np.sqrt(obs_var_fit)
            # apply K-correction
            obs_model_flux = self.sed.model.bandflux(band, pred_phase, zp=zp[0], zpsys=zpsys[0])
            obs_kcorr_flux = obs_model_flux * obs_ratio_fit
            obs_kcorr_error = obs_model_flux * obs_std_fit
            
            ####################
            # rest-frame model #
            ####################
            pred_rest_wave = np.array([eff_wave * (1 + self.z)] * len(pred_phase))
            rest_ratio_fit, rest_var_fit = self.gp_predict(pred_phase, pred_rest_wave)
            rest_std_fit = np.sqrt(rest_var_fit)
            # apply K-correction
            rest_model_flux = self.sed.rest_model.bandflux(band, pred_phase, zp=zp[0], zpsys=zpsys[0])
            rest_kcorr_flux = rest_model_flux * rest_ratio_fit
            rest_kcorr_error = rest_model_flux * rest_std_fit

            ########
            # Plot #
            ########
            colour = self.sed.colours[band]
            # data
            ax[0].errorbar(phase, flux, flux_err, ls="", marker="o", color=colour, label=band)
            # model
            #ax[0].plot(pred_phase, rest_kcorr_flux, color=colour, ls='dotted')
            ax[0].plot(pred_phase, obs_kcorr_flux, color=colour)
            ax[0].fill_between(pred_phase, 
                               obs_kcorr_flux - rest_kcorr_error, 
                               obs_kcorr_flux + rest_kcorr_error, 
                               alpha=0.2,
                               color=colour)
        
            # residuals
            norm = np.average(ratio_flux, weights=1 / ratio_error ** 2)  # for plotting only
            # ratio
            ax[1].errorbar(phase, ratio_flux / norm, ratio_error / norm, 
                           ls="", marker="o", color=colour)
            # fit
            ax[1].plot(pred_phase, obs_ratio_fit / norm, color=colour)
            ax[1].fill_between(pred_phase, 
                               (obs_ratio_fit - obs_std_fit) / norm, 
                               (obs_ratio_fit + obs_std_fit) / norm, 
                               alpha=0.2, color=colour)
            
            # config
            ax[0].set_ylabel(r'$F_{\lambda}$', fontsize=16)
            ax[1].set_xlabel('Days since B-maximum', fontsize=16)
            ax[1].set_ylabel(r'$F_{\lambda}^{\rm data} / F_{\lambda}^{\rm SED}$', fontsize=16)
            ax[0].set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
            for i in range(2):
                ax[i].tick_params('both', labelsize=14)
        ax[0].legend(fontsize=14)
        plt.show()

    def calculate_colour(self, band1: str, band2: str, zp: float = 30, zpsys: str = 'ab', 
                         plot: bool = True):
        """Calculates rest-frame colour using the colour-matched SED.

        Note: Colour = band1 - band2
        
        Parameters
        ----------
        band1: First band.
        band2: Second band.
        zp: Zeropoint for both bands. Only used if the photometry does not 
            include any of the given bands.
        zpsys: Magnitude system for both bands. Only used if the photometry
            does not  include any of the given bands.
        plot: Whether to plot the colour curve.

        Results
        -------
        colour: Colour curve.
        colour_err: Uncertainty.
        """
        # phase range to use
        minphase, maxphase = self.phase_range
        pred_phase = np.arange(minphase, maxphase + 0.1, 0.1)
        self.pred_phase = pred_phase

        if (band1 not in self.phot.band) | (band2 not in self.phot.band):
            eff_wave1 = sncosmo.get_bandpass(band1).wave_eff
            eff_wave2 = sncosmo.get_bandpass(band2).wave_eff
            zp1 = zp2 = zp
            zpsys1 = zpsys2 = zpsys
        else:
            # create band and phase mask
            band_mask1 = self.phot.band == band1
            band_mask2 = self.phot.band == band2
            mask1 = self.phase_mask & band_mask1
            mask2 = self.phase_mask & band_mask2
            # apply mask
            eff_wave1 = self.phot.eff_wave[mask1][0]
            eff_wave2 = self.phot.eff_wave[mask2][0]
            zp1 = self.phot.zp[mask1][0]
            zp2 = self.phot.zp[mask2][0]
            zpsys1 = self.phot.zpsys[mask1][0]
            zpsys2 = self.phot.zpsys[mask2][0]
        
        # wavelength array
        pred_wave = np.array([eff_wave1 * (1 + self.z)] * len(pred_phase) + 
                             [eff_wave2 * (1 + self.z)] * len(pred_phase) 
                            )
        # flux array
        rest_model_flux1 = self.sed.rest_model.bandflux(band1, pred_phase, zp=zp1, zpsys=zpsys1)
        rest_model_flux2 = self.sed.rest_model.bandflux(band2, pred_phase, zp=zp2, zpsys=zpsys2)
        rest_model_flux = np.r_[rest_model_flux1, rest_model_flux2]
        # K-corr. predict
        pred_phase_ = np.r_[pred_phase, pred_phase]
        ratio_fit, cov_fit = self.gp_predict(pred_phase_, pred_wave, return_cov=True)
        rest_kcorr_flux = rest_model_flux * ratio_fit
        rest_kcorr_cov = np.outer(rest_model_flux, rest_model_flux) * cov_fit
        self.colour_flux_ratio = rest_kcorr_flux
        self.colour_flux_cov = rest_kcorr_cov
        
        # compute flux ratio for the colour
        colour, colour_err = self._compute_colour(rest_kcorr_flux, rest_kcorr_cov, zp1, zp2)
        self.colour, self.colour_err = colour, colour_err
        self._compute_colour_stretch()

        if plot is True:
            fig, ax = plt.subplots()
            ax.plot(pred_phase, colour)
            ax.fill_between(pred_phase, colour - colour_err, colour + colour_err, 
                            alpha=0.2)
            ax.set_ylabel(fr'$({band1} - {band2})$ (mag)', fontsize=16)
            ax.set_xlabel('Days since B-maximum', fontsize=16)
            ax.set_title(f'"{self.source}" SED source (z={self.z})', fontsize=16)
            ax.tick_params('both', labelsize=14)
            plt.show()

    def _compute_colour(self, fluxes: np.ndarray, cov: np.ndarray, 
                        zp1: float | np.ndarray, zp2: float | np.ndarray):
        """Computes the colour curve from concatenated flux arrays from two bands.

        Note: flux = [flux1_0, flux1_1, ...flux1_N, # band1
                      flux2_N+1, flux2_N+2, ...flux2_2N]  # band2
        Parameters
        ----------
        flux: Flux from two bands.
        cov: Covariance from two bands.
        zp1: Zero point of first band.
        zp2: Zero point of second band.
        """
        N = fluxes.shape[0] // 2
        f1 = fluxes[:N]
        f2 = fluxes[N:]
        # variance and covariance
        cov_11 = cov[:N, :N]
        cov_22 = cov[N:, N:]
        cov_12 = cov[:N, N:]
        
        # error propagation
        prefactor = 2.5 / np.log(10)
        var_colour = (
            (np.diag(cov_11) / (f1 ** 2)) +
            (np.diag(cov_22) / (f2 ** 2)) -
            2 * np.diag(cov_12) / (f1 * f2)
        ) * (prefactor ** 2)

        colour = -2.5 * np.log10(f1 / f2) + (zp1 - zp2)
        colour_err = np.sqrt(var_colour)
    
        return colour, colour_err

    def _compute_colour_stretch(self):
        # colour-stretch between 0.4 and 1.4 translate to phases between 12 and 42 days
        # assuming sBV...
        mask = (12 < self.pred_phase) & (self.pred_phase < 42)  
        pred_phase = self.pred_phase[mask]
        mask = np.array(list(mask) + list(mask))
        fluxes = np.random.multivariate_normal(self.colour_flux_ratio[mask], 
                                               self.colour_flux_cov[np.ix_(mask, mask)], 
                                               size=1000)

        # calculate mean and std through monte-carlo sampling
        st_list = []
        for flux in fluxes:
            N = len(flux) // 2
            f1 = flux[:N]
            f2 = flux[N:]
            flux_ratio = -2.5 * np.log10(f1 / f2)
            st_idx = np.argmax(flux_ratio)
            st_list.append(pred_phase[st_idx])
        self.st, self.st_err = np.mean(st_list) / 30, np.std(st_list) / 30