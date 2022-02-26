import matplotlib.pyplot as plt
import numpy as np


def plotHistogram(arrayPerAmp, showMasked=False, bins=20, **config):
    """ """
    # default clipping config
    clippingConfig = dict(method='iqr', sigma=3)
    clippingConfig.update(**config)

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))

    for ai, amp in enumerate(arrayPerAmp):
        ax = axs[ai // 2, ai % 2]
        # masked = ccdStats.sigmaClip(amp.ravel(), axis=0, **clippingConfig)
        mean, sig = round(np.mean(amp), 3), round(np.std(amp), 3)
        ax.hist(amp.ravel(), bins=bins, label=f'level : {mean} +-{sig}')
        if showMasked:
            ax.hist(amp[amp.mask].ravel(), bins=bins)
        ax.set_ylabel(f'amp{ai}')
        # ax.set_xlim(mean - 5 * sig, mean + 5 * sig)
        ax.grid()
        ax.legend()
