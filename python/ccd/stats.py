import numpy as np
from astropy.stats import sigma_clip


def crudePolyfit(y, deg, x=None):
    """ return fitted polynomial given y and degree."""
    x = np.arange(len(y)) if x is None else x
    p = np.polyfit(x, y, deg=deg)
    return np.polyval(p, x)


def sigmaClip(array, axis, clippingMethod='iqr', sigma=3):
    """ return sigma-clipped array for given axis, clippingMethod and sigma."""
    knownMethods = ['iqr', 'sigma']

    if clippingMethod == 'iqr':
        return iqrClipping(array, sigma, axis)
    elif clippingMethod == 'sigma':
        return sigma_clip(array, sigma=sigma, axis=axis, maxiters=10)
    else:
        raise ValueError(f'{clippingMethod} not in {knownMethods}')


def iqrClipping(array, sigma, axis):
    """ return masked array from IQR sigma-clipping """
    iqrToStd = 0.7413
    lowerQuartile, medianBiasArr, upperQuartile = np.percentile(array, [25.0, 50.0, 75.0], axis=axis)
    stdevBiasArr = iqrToStd * (upperQuartile - lowerQuartile)  # robust stdev
    # expand axis if necessary
    if axis > 0:
        medianBiasArr = np.expand_dims(medianBiasArr, axis=axis)
        stdevBiasArr = np.expand_dims(stdevBiasArr, axis=axis)

    diff = np.abs(array - medianBiasArr)
    maskedArray = np.ma.masked_where(diff > sigma * stdevBiasArr, array)
    return maskedArray


def clippedMean(array, axis, clippingMethod, sigma):
    """ return sigma-clipped mean for given axis, clippingMethod and sigma"""
    return sigmaClip(array, axis, clippingMethod=clippingMethod, sigma=sigma).mean(axis=axis)


def merge(array, axis, mergingMethod, **clippingConfig):
    """ return a merged cube given a mergingMethod."""
    knownMethods = ['median', 'clippedMean']

    if mergingMethod == 'median':
        return np.median(array, axis=axis)
    elif mergingMethod == 'clippedMean':
        return clippedMean(array, axis=axis, **clippingConfig).data
    else:
        raise ValueError(f'{mergingMethod} not in {knownMethods}')
