from importlib import reload

import ccd.overscan as overscan
import ccd.stats as ccdStats
import matplotlib.pyplot as plt
import numpy as np
from fpga import geom

reload(ccdStats)
reload(overscan)


class amps(object):
    nRows = 4176
    nCols = 512
    nAmps = 8

    colTrim = (0, 0)
    rowTrim = (0, 0)

    serialOS = overscan.SerialOS()

    mergeColumnConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)
    mergeRowConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)

    superFrameMergeConfig = dict(mergingMethod='clippedMean', clippingMethod='iqr', sigma=3)

    superBias = None
    superDark = None

    subOverscan = True
    subSuperBias = False
    subSuperDark = False

    boxSize = 100

    def split(self, im, rowTrim=None, colTrim=None, osColTrim=None, subOverscan=None, subSuperBias=None,
              subSuperDark=None):
        """ split amps and subtract overscan"""
        exp = geom.Exposure(im)
        ampIms, serialOs, _ = exp.splitImage()
        # convert to array
        ampIms = np.array(ampIms).astype('f4')
        serialOs = np.array(serialOs).astype('f4')
        # default argument to class attribute.
        rowTrim = self.rowTrim if rowTrim is None else rowTrim
        colTrim = self.colTrim if colTrim is None else colTrim
        osColTrim = self.serialOS.colTrim if osColTrim is None else osColTrim
        # select rows/cols
        rowStart, rowEnd = rowTrim[0], amps.nRows - rowTrim[1]
        colStart, colEnd = colTrim[0], amps.nCols - colTrim[1]
        osColStart, osColEnd = osColTrim[0], amps.serialOS.nCols - osColTrim[1]
        # final cutoff.
        ampIms = ampIms[:, rowStart:rowEnd, colStart:colEnd]
        serialOs = serialOs[:, rowStart:rowEnd, osColStart:osColEnd]

        subOverscan = self.subOverscan if subOverscan is None else subOverscan
        subSuperBias = self.subSuperBias if subSuperBias is None else subSuperBias
        subSuperDark = self.subSuperDark if subSuperDark is None else subSuperDark

        if subOverscan:
            biasLevelPerRow = self.serialOS.levelPerRow(serialOs)
            ampIms -= biasLevelPerRow[:, :, None]

        if subSuperBias:
            superBias = np.zeros(ampIms.shape) if self.superBias is None else self.superBias
            ampIms -= superBias

        if subSuperDark:
            exptime = float(exp.expTime)
            superDark = np.zeros(ampIms.shape) if self.superDark is None else self.superDark
            ampIms -= exptime * superDark

        return ampIms, serialOs

    def combine(self, frames):
        """ """

        def getAmps(frame):
            ampIms, osIms = self.split(frame)
            return ampIms

        return np.array([getAmps(frame) for frame in frames])

    def superFrame(self, frames, superFrameMergeConfig=None):
        """ """
        # combine all frames
        cubeArray = self.combine(frames)
        # retrieve superFrame argument.
        superFrameMergeConfig = self.superFrameMergeConfig if superFrameMergeConfig is None else superFrameMergeConfig
        # merge cube
        return ccdStats.merge(cubeArray, axis=0, **superFrameMergeConfig)

    def setSuperBias(self, superBias):
        """ """
        print('setting new superBias')
        self.superBias = superBias
        print('activating superBias subtraction')
        self.subSuperBias = True

    def setSuperDark(self, superDark, darktime):
        """ """
        print(f'setting new superDark {darktime} s')
        # scaling superDark
        superDark /= darktime
        self.superDark = superDark
        print('activating superDark subtraction')
        self.subSuperDark = True

    #
    # def signalPerAmp(self, frames, sigma=5):
    #     """ """
    #     signal = []
    #
    #     for frame in frames:
    #         ampIms, serialOs = self.split(frame)
    #         signal.append(ccdStats.clippedMeanPerAmp(ampIms, sigma=sigma))
    #
    #     return pd.DataFrame(np.array(signal), columns=[f'amp{ampId}' for ampId in range(self.nAmps)])

    # @staticmethod
    # def plotLevelPerRow(arrayPerAmp, **config):
    #     """ """
    #     # default clipping config
    #     clippingConfig = dict(method='iqr', sigma=3)
    #     clippingConfig.update(**config)
    #
    #     fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
    #     clippedPerRow = ccdStats.clippedMean(arrayPerAmp, axis=2, **clippingConfig)
    #
    #     for ai, amp in enumerate(clippedPerRow):
    #         axs[ai].set_ylabel(f'amp{ai}')
    #         axs[ai].plot(amp, alpha=0.5, label=f'clippedMean {clippingConfig["sigma"]}-sigma')
    #         axs[ai].set_xlim(-100, 5200)
    #         axs[ai].grid()
    #         axs[ai].legend()
    #
    #     return axs
    #
    # @staticmethod
    # def histLevelPerRow(arrayPerAmp, **config):
    #     """ """
    #     # default clipping config
    #     clippingConfig = dict(method='iqr', sigma=3)
    #     clippingConfig.update(**config)
    #
    #     fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(14, 8), sharex=True)
    #     clippedPerRow = ccdStats.clippedMean(arrayPerAmp, axis=2, **clippingConfig)
    #
    #     for ai, amp in enumerate(clippedPerRow):
    #         axs[ai].set_ylabel(f'amp{ai}')
    #         axs[ai].plot(amp, alpha=0.5, label=f'clippedMean {clippingConfig["sigma"]}-sigma')
    #         axs[ai].set_xlim(-100, 5200)
    #         axs[ai].grid()
    #         axs[ai].legend()
    #
    #     return axs
    #
    # @staticmethod
    # def plotHistogram(arrayPerAmp, bins=20, **config):
    #     """ """
    #     # default clipping config
    #     clippingConfig = dict(method='iqr', sigma=3)
    #     clippingConfig.update(**config)
    #
    #     fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
    #
    #     for ai, amp in enumerate(arrayPerAmp):
    #         ax = axs[ai // 2, ai % 2]
    #         masked = ccdStats.sigmaClip(amp.ravel(), axis=0, **clippingConfig)
    #         mean, sig = round(np.mean(masked), 3), round(np.std(masked), 3)
    #         ax.hist(masked.ravel(), bins=bins, label=f'level : {mean} +-{sig}')
    #         ax.set_ylabel(f'amp{ai}')
    #         ax.set_xlim(mean - 5 * sig, mean + 5 * sig)
    #         ax.grid()
    #         ax.legend()
    #
    # def ampRois(self, frame, boxSize=None):
    #     """ """
    #     boxSize = self.boxSize if boxSize is None else boxSize
    #     ampIms, osIms = self.split(frame)
    #
    #     nRowsRoi = amps.nRows // boxSize
    #     nColsRoi = amps.nCols // boxSize
    #     nRois = nRowsRoi * nColsRoi
    #
    #     trimRows = int((amps.nRows % boxSize) / 2)
    #     trimCols = int((amps.nCols % boxSize) / 2)
    #
    #     ampRois = np.zeros((amps.nAmps, nRois, boxSize, boxSize))
    #
    #     for iAmp, ampArray in enumerate(ampIms):
    #         trimmed = ampArray[trimRows:-trimRows, trimCols:-trimCols]
    #
    #         for iRow in range(nRowsRoi):
    #             for iCol in range(nColsRoi):
    #                 iBox = iRow * nColsRoi + iCol
    #                 ampRois[iAmp, iBox] = trimmed[iRow * boxSize:iRow * boxSize + boxSize,
    #                                       iCol * boxSize:iCol * boxSize + boxSize]
    #
    #     return ampRois
