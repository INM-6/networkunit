import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def power_spectral_density(sample1, freqs1, sample2=None, freqs2=None,
                           ax=None, palette=None,
                           sample_names=['observation', 'prediction'],
                           **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if palette is None:
        palette = [sns.color_palette()[0], sns.color_palette()[1]]

    ax.plot(freqs1, np.squeeze(sample1), color=palette[0], label=sample_names[0])

    if sample2 is not None:
        if freqs2 is None:
            freqs2 = freqs1
        ax.plot(freqs2, np.squeeze(sample2), color=palette[1], label=sample_names[1])

    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('power spectral density')
    ax.set_yscale('log')
    return ax