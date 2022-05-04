from importlib import reload

import ccd.stats as ccdStats
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

reload(ccdStats)


def HCTE(x, a, b, c, d, bias):
    return a / x ** 4 + b / x ** 3 + c / x ** 2 + d / x + bias


class SerialOS(object):
    colTrim = (0, 0)
    nCols = 32

    # sigma is default to 3 given the number of overscan column after trimming (31),
    mergeColumnConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
    mergeRowConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
    mergeAmpConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)

    # model overscan per row.
    modelPerRow = True

    # polyfit deg is set to 3, no need to overfit which would increase the error.
    fitPolynomial = True
    polyfitConfig = dict(deg=3, nIter=5, clippingMethod='iqr', sigma=3)

    def levelPerColumn(self, osIms, subMedian=False, doPlot=False, **updateMergeRowConfig):
        """ plot serial level per amp and per column, useful to know which columns to trim."""
        # update mergeRowConfig
        mergeRowConfig = self.mergeRowConfig
        mergeRowConfig.update(updateMergeRowConfig)

        levelPerColumn = ccdStats.merge(osIms, axis=1, **mergeRowConfig)

        if subMedian:
            levelPerColumn -= np.median(levelPerColumn, axis=1)[:, np.newaxis]

        if doPlot:
            plt.figure(figsize=(12, 6))
            plt.title(str(mergeRowConfig))

            for ai, perColumn in enumerate(levelPerColumn):
                plt.plot(perColumn, label=f'amp{ai}')

            plt.grid()
            plt.legend()

        return levelPerColumn

    def modelHCTE(self, osIms, **updateMergeRowConfig):
        """ model extra counts from HCTE."""

        levelPerColumn = self.levelPerColumn(osIms, **updateMergeRowConfig)
        x = np.arange(levelPerColumn.shape[1]) + 1
        correction = np.zeros(levelPerColumn.shape)

        for iAmp in range(osIms.shape[0]):
            [a, b, c, d, bias], pcov = curve_fit(HCTE, x, levelPerColumn[iAmp])
            correction[iAmp] = HCTE(x, *[a, b, c, d, 0])

        return correction

    def levelPerRow(self, osIms, subMedian=False, doPlot=False, **updateMergeColumnConfig):
        """ plot serial level per amp and per column, useful to know which columns to trim."""
        # update mergeRowConfig
        mergeColumnConfig = self.mergeColumnConfig
        mergeColumnConfig.update(updateMergeColumnConfig)

        levelPerRow = ccdStats.merge(osIms, axis=2, **mergeColumnConfig)

        if subMedian:
            levelPerRow -= np.median(levelPerRow, axis=1)[:, np.newaxis]

        if doPlot:
            fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
            axs[0].set_title(str(mergeColumnConfig))

            for ai, amp in enumerate(levelPerRow):
                axs[ai].plot(amp, alpha=0.5, label=f'amp{ai}')
                axs[ai].set_xlim(-100, 4700)
                axs[ai].grid()
                axs[ai].legend()

        return levelPerRow

    def polyfit(self, levelPerRow, deg, nIter=5, doPlot=False, **residRejectionConfig):
        """ """
        models = np.ones(levelPerRow.shape)
        residuals = np.ma.masked_array(levelPerRow.copy())

        for ai, amp in enumerate(levelPerRow):
            # for each amp
            rows = np.arange(len(amp))
            masked = np.ma.masked_array(amp)
            for it in range(nIter):
                # fit a polynomial with a masked array.
                p = np.ma.polyfit(rows, masked, deg=deg)
                model = np.polyval(p, rows)
                resid = masked - model
                # update the mask from the sigma-clipping of the residuals.
                newMask = ccdStats.sigmaClip(resid, axis=0, **residRejectionConfig)
                masked.mask = newMask.mask

            # you want all the "final" data.
            resid = np.ma.masked_array(amp - model)
            resid.mask = newMask.mask

            models[ai] = model
            residuals[ai] = resid

        if doPlot:
            fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
            axs[0].set_title(f'polyfit(deg={deg}) {str(residRejectionConfig)}')

            for ai, amp in enumerate(levelPerRow):
                axs[ai].plot(amp, alpha=0.5, label=f'amp{ai}')
                axs[ai].plot(models[ai])
                axs[ai].set_xlim(-100, 4700)
                axs[ai].grid()
                axs[ai].legend()

        return models, residuals

    def estimate(self, osIms, modelPerRow=None, fitPolynomial=None, updatePolyfitConfig=None,
                 updateMergeColumnConfig=None, updateMergeAmpConfig=None):
        """ estimate a robust signal value per row"""

        def updateConfig(config, update):
            if update is not None:
                config.update(update)
            return config

        # get and update config
        modelPerRow = self.modelPerRow if modelPerRow is None else modelPerRow
        fitPolynomial = self.fitPolynomial if fitPolynomial is None else fitPolynomial
        polyfitConfig = updateConfig(self.polyfitConfig, updatePolyfitConfig)
        mergeColumnConfig = updateConfig(self.mergeColumnConfig, updateMergeColumnConfig)
        mergeAmpConfig = updateConfig(self.mergeAmpConfig, updateMergeAmpConfig)

        # model your overscan per-amp and per-row.
        if modelPerRow:
            levelPerRow = self.levelPerRow(osIms, **mergeColumnConfig)
            if fitPolynomial:
                model, residuals = self.polyfit(levelPerRow, **polyfitConfig)
            else:
                model = levelPerRow
        # just a single value per-amp
        else:
            model = ccdStats.merge(osIms, axis=(1, 2), **mergeAmpConfig)

        # keep same dimensions.
        expandDim = 2 if modelPerRow else (1, 2)

        return np.expand_dims(model, expandDim)
