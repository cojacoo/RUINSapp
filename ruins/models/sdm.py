# BIAS CORRECTION
import numpy as np
import pandas as pd
from scipy.stats import gamma
from scipy.stats import norm
from scipy.signal import detrend

'''
Scaled distribution mapping for climate data

This is a excerpt from pyCAT and the method after Switanek et al. (2017) containing the functions to perform a relative and absolute bias correction on climate data.
(cc) c.jackisch@tu-braunschweig.de, July 2020

It is intended to be used on pandas time series at single locations/pixels.

Switanek, M. B., P. A. Troch, C. L. Castro, A. Leuprecht, H.-I. Chang, R. Mukherjee, and E. M. C. Demaria (2017), Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes, Hydrol. Earth Syst. Sci., 21(6), 2649–2666, https://doi.org/10.5194/hess-21-2649-2017
'''


def relSDM(obs, mod, sce, cdf_threshold=0.9999999, lower_limit=0.1):
    '''relative scaled distribution mapping assuming a gamma distributed parameter (with lower limit zero)
    rewritten from pyCAT for 1D data

    obs :: observed variable time series
    mod :: modelled variable for same time series as obs
    sce :: to unbias modelled time series
    cdf_threshold :: upper and lower threshold of CDF
    lower_limit :: lower limit of data signal (values below will be masked!)

    returns corrected timeseries
    tested with pandas series.
    '''

    obs_r = obs[obs >= lower_limit]
    mod_r = mod[mod >= lower_limit]
    sce_r = sce[sce >= lower_limit]

    obs_fr = 1. * len(obs_r) / len(obs)
    mod_fr = 1. * len(mod_r) / len(mod)
    sce_fr = 1. * len(sce_r) / len(sce)
    sce_argsort = np.argsort(sce)

    obs_gamma = gamma.fit(obs_r, floc=0)
    mod_gamma = gamma.fit(mod_r, floc=0)
    sce_gamma = gamma.fit(sce_r, floc=0)

    obs_cdf = gamma.cdf(np.sort(obs_r), *obs_gamma)
    mod_cdf = gamma.cdf(np.sort(mod_r), *mod_gamma)
    obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
    mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold

    expected_sce_raindays = min(int(np.round(len(sce) * obs_fr * sce_fr / mod_fr)), len(sce))
    sce_cdf = gamma.cdf(np.sort(sce_r), *sce_gamma)
    sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

    # interpolate cdf-values for obs and mod to the length of the scenario
    obs_cdf_intpol = np.interp(np.linspace(1, len(obs_r), len(sce_r)), np.linspace(1, len(obs_r), len(obs_r)), obs_cdf)
    mod_cdf_intpol = np.interp(np.linspace(1, len(mod_r), len(sce_r)), np.linspace(1, len(mod_r), len(mod_r)), mod_cdf)

    # adapt the observation cdfs
    obs_inverse = 1. / (1 - obs_cdf_intpol)
    mod_inverse = 1. / (1 - mod_cdf_intpol)
    sce_inverse = 1. / (1 - sce_cdf)
    adapted_cdf = 1 - 1. / (obs_inverse * sce_inverse / mod_inverse)
    adapted_cdf[adapted_cdf < 0.] = 0.

    # correct by adapted observation cdf-values
    xvals = gamma.ppf(np.sort(adapted_cdf), *obs_gamma) * gamma.ppf(sce_cdf, *sce_gamma) / gamma.ppf(sce_cdf,
                                                                                                     *mod_gamma)

    # interpolate to the expected length of future raindays
    correction = np.zeros(len(sce))
    if len(sce_r) > expected_sce_raindays:
        xvals = np.interp(np.linspace(1, len(sce_r), expected_sce_raindays), np.linspace(1, len(sce_r), len(sce_r)),
                          xvals)
    else:
        xvals = np.hstack((np.zeros(expected_sce_raindays - len(sce_r)), xvals))

    correction[sce_argsort[-expected_sce_raindays:]] = xvals

    return pd.Series(correction, index=sce.index)


def absSDM(obs, mod, sce, cdf_threshold=0.9999999):
    '''absolute scaled distribution mapping assuming a normal distributed parameter
    rewritten from pyCAT for 1D data

    obs :: observed variable time series
    mod :: modelled variable for same time series as obs
    sce :: to unbias modelled time series
    cdf_threshold :: upper and lower threshold of CDF

    returns corrected timeseries
    tested with pandas series.
    '''

    obs_len = len(obs)
    mod_len = len(mod)
    sce_len = len(sce)
    obs_mean = np.mean(obs)
    mod_mean = np.mean(mod)
    smean = np.mean(sce)
    odetrend = detrend(obs)
    mdetrend = detrend(mod)
    sdetrend = detrend(sce)

    obs_norm = norm.fit(odetrend)
    mod_norm = norm.fit(mdetrend)
    sce_norm = norm.fit(sdetrend)

    sce_diff = sce - sdetrend
    sce_argsort = np.argsort(sdetrend)

    obs_cdf = norm.cdf(np.sort(odetrend), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mdetrend), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sdetrend), *sce_norm)
    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    # interpolate cdf-values for obs and mod to the length of the scenario
    obs_cdf_intpol = np.interp(np.linspace(1, obs_len, sce_len), np.linspace(1, obs_len, obs_len), obs_cdf)
    mod_cdf_intpol = np.interp(np.linspace(1, mod_len, sce_len), np.linspace(1, mod_len, mod_len), mod_cdf)

    # adapt the observation cdfs
    # split the tails of the cdfs around the center
    obs_cdf_shift = obs_cdf_intpol - .5
    mod_cdf_shift = mod_cdf_intpol - .5
    sce_cdf_shift = sce_cdf - .5
    obs_inverse = 1. / (.5 - np.abs(obs_cdf_shift))
    mod_inverse = 1. / (.5 - np.abs(mod_cdf_shift))
    sce_inverse = 1. / (.5 - np.abs(sce_cdf_shift))
    adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
    adapted_cdf[adapted_cdf < 0] += 1.
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) \
            + obs_norm[-1] / mod_norm[-1] \
            * (norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))
    xvals -= xvals.mean()
    xvals += obs_mean + (smean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals
    correction += sce_diff - smean

    return correction


def SDM(obs, mod, sce, meth='rel', cdf_threshold=0.9999999, lower_limit=0.1):
    '''scaled distribution mapping - wrapper to relative and absolute bias correction functions
    rewritten from pyCAT for 1D data

    obs :: observed variable time series
    mod :: modelled variable for same time series as obs
    sce :: to unbias modelled time series
    meth :: 'rel' for relative SDM, else absolute SDM will be performed
    cdf_threshold :: upper and lower threshold of CDF
    lower_limit :: lower limit of data signal (values below will be masked when meth != 'rel')

    The original authors suggest to use the absolute SDM for air temperature and the relative SDM for precipitation and radiation series.

    returns corrected timeseries
    tested with pandas series.
    '''

    if meth == 'rel':
        return relSDM(obs, mod, sce, cdf_threshold, lower_limit)
    else:
        return absSDM(obs, mod, sce, cdf_threshold)