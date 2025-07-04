import types
from sncosmo import get_bandpass, get_magsystem
from sncosmo.utils import integration_grid
from sncosmo.models import _check_for_fitpack_error
from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING

def kcorr_flux(self, phase, wave, z, gp_predict):
    phase = np.array(phase)
    wave = np.array(wave)
    if np.any(wave < self.minwave()) or np.any(wave > self.maxwave()):
        raise ValueError('requested wavelength value(s) outside '
                         'model range')
    try:
        f = self._flux(phase, wave)
    except ValueError as e:
        _check_for_fitpack_error(e, phase, 'phase')
        _check_for_fitpack_error(e, wave, 'wave')
        raise e

    ########
    # k-corr
    phase_ = np.array([phase] * len(wave))
    ratio_fit, var_fit = gp_predict(phase_, wave * (1 + z))
    f *= ratio_fit
    ########
    
    if phase.ndim == 0:
        if wave.ndim == 0:
            return f[0, 0]
        return f[0, :]
    return f
    
###########################
def _kcorr_bandflux_single(model, band, time_or_phase, z, gp_predict):
    if (band.minwave() < model.minwave() or band.maxwave() > model.maxwave()):
        raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                         'outside spectral range [{3:.6g}, .., {4:.6g}]'
                         .format(band.name, band.minwave(), band.maxwave(),
                                 model.minwave(), model.maxwave()))

    # Set up wavelength grid. Spacing (dwave) evenly divides the bandpass,
    # closest to 5 angstroms without going over.
    wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                   MODEL_BANDFLUX_SPACING)
    trans = band(wave)
    f = model._flux(time_or_phase, wave)

    ########
    # k-corr
    f_ = np.empty(0)
    for flux in f:
        f_ = np.r_[f_, flux]
    PHASE, WAVE = np.meshgrid(time_or_phase, wave, indexing='ij')
    pairs = np.stack([PHASE.ravel(), WAVE.ravel()], axis=1)
    phase_, wave_ = pairs.T
    ratio_fit, var_fit = gp_predict(phase_, wave_ * (1 + z))
    f_err_ = f_ * np.sqrt(var_fit)
    f_ *= ratio_fit
    
    f = f_.reshape(len(time_or_phase), len(wave))
    f_err = f_err_.reshape(len(time_or_phase), len(wave))
    ########

    fsum = np.sum(wave * trans * f, axis=1) * dwave / HC_ERG_AA
    fsum_err = np.sum(wave * trans * f_err, axis=1) * dwave / HC_ERG_AA
    return fsum, fsum_err

def _kcorr_bandflux(model, band, time_or_phase, zp, zpsys, z, gp_predict):
    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')

    # broadcast arrays
    if zp is None:
        time_or_phase, band = np.broadcast_arrays(time_or_phase, band)
    else:
        time_or_phase, band, zp, zpsys = \
            np.broadcast_arrays(time_or_phase, band, zp, zpsys)

    # Convert all to 1-d arrays.
    ndim = time_or_phase.ndim  # Save input ndim for return val.
    time_or_phase = np.atleast_1d(time_or_phase)
    band = np.atleast_1d(band)
    if zp is not None:
        zp = np.atleast_1d(zp)
        zpsys = np.atleast_1d(zpsys)

    # initialize output arrays
    bandflux = np.zeros(time_or_phase.shape, dtype=float)
    bandflux_err = np.zeros(time_or_phase.shape, dtype=float)

    # Loop over unique bands.
    for b in set(band):
        mask = band == b
        b = get_bandpass(b)

        fsum, fsum_err = _kcorr_bandflux_single(model, b, time_or_phase[mask], 
                                      z, gp_predict)
        
        if zp is not None:
            zpnorm = 10.**(0.4 * zp[mask])
            bandzpsys = zpsys[mask]
            for ms in set(bandzpsys):
                mask2 = bandzpsys == ms
                ms = get_magsystem(ms)
                zpnorm[mask2] = zpnorm[mask2] / ms.zpbandflux(b)
            fsum *= zpnorm
            fsum_err *= zpnorm

        bandflux[mask] = fsum
        bandflux_err[mask] = fsum_err

    if ndim == 0:
        return bandflux[0], bandflux_err[0]
    return bandflux, bandflux_err
    
def kcorr_bandflux(self, band, phase, zp, zpsys, z, gp_predict):
    try:
        return _kcorr_bandflux(self, band, phase, zp, zpsys, 
                               z, gp_predict)
    except ValueError as e:
        _check_for_fitpack_error(e, phase, 'phase')
        raise e

# update flux
new_fluc = partial(kcorr_flux, z=0.0, gp_predict=kcorr.gp_predict)
kcorr.sed.model.flux =  types.MethodType(new_fluc, kcorr.sed.model)
new_fluc = partial(kcorr_flux, z=kcorr.z, gp_predict=kcorr.gp_predict)
kcorr.sed.rest_model.flux =  types.MethodType(new_fluc, kcorr.sed.rest_model)

# update bandflux
new_fluc = partial(kcorr_bandflux, z=0.0, gp_predict=kcorr.gp_predict)
kcorr.sed.model.kcorr_bandflux =  types.MethodType(new_fluc, kcorr.sed.model)
new_fluc = partial(kcorr_bandflux, z=kcorr.z, gp_predict=kcorr.gp_predict)
kcorr.sed.rest_model.kcorr_bandflux =  types.MethodType(new_fluc, kcorr.sed.model)