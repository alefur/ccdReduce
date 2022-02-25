from importlib import reload

import ccd.stats as ccdStats
import matplotlib.pyplot as plt
import numpy as np

reload(ccdStats)


class SerialOS(object):
    colTrim = (0, 0)
    nCols = 32

    # sigma is default to 3 given the number of overscan column after trimming (31),
    mergeColumnConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
    mergeRowConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)

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

    def modelLevelPerRow(self, osIms, doPlot=False, fitPolynomial=None, polyfitConfig=None, mergeColumnConfig=None):
        """ estimate a robust signal value per row"""
        fitPolynomial = self.fitPolynomial if fitPolynomial is None else fitPolynomial
        polyfitConfig = self.polyfitConfig if polyfitConfig is None else polyfitConfig
        mergeColumnConfig = self.mergeColumnConfig if mergeColumnConfig is None else mergeColumnConfig

        levelPerRow = self.levelPerRow(osIms, **mergeColumnConfig)

        if fitPolynomial:
            model = self.polyfit(levelPerRow, **polyfitConfig)
        else:
            model = levelPerRow

        return model

    def polyfit(self, levelPerRow, deg, nIter=5, **residRejectionConfig):
        """ """
        models = np.ones(levelPerRow.shape)

        for ai, amp in enumerate(levelPerRow):
            rows = np.arange(len(amp))
            masked = np.ma.masked_array(amp)
            for it in range(nIter):
                p = np.ma.polyfit(rows, masked, deg=deg)
                model = np.polyval(p, rows)
                resid = masked - model
                newMask = ccdStats.sigmaClip(resid, axis=0, **residRejectionConfig)
                masked.mask = newMask.mask

            models[ai] = model

        return models

    # def levelPerRow(self, arrayPerAmp, fitPolynomial=None, doPlot=False, polyfitConfig=None, clippingConfig=None):
    #     """ estimate a robust signal value per row"""
    #     # default argument from class attribute
    #     fitPolynomial = self.fitPolynomial if fitPolynomial is None else fitPolynomial
    #     polyfitConfig = self.polyfitConfig if polyfitConfig is None else polyfitConfig
    #     clippingConfig = self.clippingConfig if clippingConfig is None else clippingConfig
    #     # empty array
    #     levels = np.zeros((arrayPerAmp.shape[0], arrayPerAmp.shape[1]), dtype='f4')
    #
    #     if doPlot:
    #         fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
    #
    #     clippedPerRow = ccdStats.clippedMean(arrayPerAmp, axis=2, **clippingConfig)
    #
    #     for ai, amp in enumerate(clippedPerRow):
    #         # sigma is default to 3 given the number of overscan column after trimming (26),
    #         # I wont necessarily increase it.
    #
    #         # polyfit deg is set to 3, no need to overfit which would increase the error.
    #         model = ccdStats.crudePolyfit(amp, **polyfitConfig)
    #         levels[ai] = model if fitPolynomial else amp
    #
    #         if doPlot:
    #             axs[ai].set_ylabel(f'amp{ai}')
    #             axs[ai].plot(amp, alpha=0.5, label=f'clippedMean {clippingConfig["sigma"]}-sigma')
    #             axs[ai].plot(model, label=f'polyfit(n={polyfitConfig["deg"]})')
    #             axs[ai].set_xlim(-100, 5200)
    #             axs[ai].grid()
    #             axs[ai].legend()
    #
    #     return levels

# from importlib import reload
#
# import ccd.stats as ccdStats
# import matplotlib.pyplot as plt
# import numpy as np
#
# reload(ccdStats)
#
#
# class SerialOS(object):
#     colTrim = (0, 0)
#     nCols = 32
#
#     mergeColumnConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
#     mergeRowConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
#
#     fitPolynomial = True
#     polyfitConfig = dict(deg=3)
#
#     def levelPerColumn(self, osIms, subMedian=False, doPlot=False, **updateMergeRowConfig):
#         """ plot serial level per amp and per column, useful to know which columns to trim."""
#         # update merging row config
#         mergeRowConfig = self.mergeRowConfig
#         mergeRowConfig.update(updateMergeRowConfig)
#
#         levelPerColumn = ccdStats.merge(osIms, axis=1, **mergeRowConfig)
#
#         plt.figure(figsize=(12, 6))
#
#         for ai, perColumn in enumerate(levelPerColumn):
#             offset = np.median(perColumn) if subMedian else 0
#             plt.plot(perColumn - offset, label=ai)
#
#         plt.grid()
#         plt.legend()
#
#     def levelPerRow(self, arrayPerAmp, fitPolynomial=None, doPlot=False, polyfitConfig=None, clippingConfig=None):
#         """ estimate a robust signal value per row"""
#         # default argument from class attribute
#         fitPolynomial = self.fitPolynomial if fitPolynomial is None else fitPolynomial
#         polyfitConfig = self.polyfitConfig if polyfitConfig is None else polyfitConfig
#         clippingConfig = self.clippingConfig if clippingConfig is None else clippingConfig
#         # empty array
#         levels = np.zeros((arrayPerAmp.shape[0], arrayPerAmp.shape[1]), dtype='f4')
#
#         if doPlot:
#             fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
#
#         clippedPerRow = ccdStats.clippedMean(arrayPerAmp, axis=2, **clippingConfig)
#
#         for ai, amp in enumerate(clippedPerRow):
#             # sigma is default to 3 given the number of overscan column after trimming (26),
#             # I wont necessarily increase it.
#
#             # polyfit deg is set to 3, no need to overfit which would increase the error.
#             model = ccdStats.crudePolyfit(amp, **polyfitConfig)
#             levels[ai] = model if fitPolynomial else amp
#
#             if doPlot:
#                 axs[ai].set_ylabel(f'amp{ai}')
#                 axs[ai].plot(amp, alpha=0.5, label=f'clippedMean {clippingConfig["sigma"]}-sigma')
#                 axs[ai].plot(model, label=f'polyfit(n={polyfitConfig["deg"]})')
#                 axs[ai].set_xlim(-100, 5200)
#                 axs[ai].grid()
#                 axs[ai].legend()
#
#         return levels
