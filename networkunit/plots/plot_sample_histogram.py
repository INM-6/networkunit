import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sample_histogram(sample1, sample2=None, ax=None, bins=100, palette=None,
                     sample_names=['observation', 'prediction'],
                     var_name='Measured Parameter', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if palette is None:
        palette = [sns.color_palette()[0], sns.color_palette()[1]]
    if sample2 is None:
        P, edges = np.histogram(sample1, bins=bins, density=True)
    else:
        if len(bins)==1:
            max_value = max([max(sample1), max(sample2)])
            min_value = min([min(sample1), min(sample2)])
            bins = np.linspace(float(min_value), float(max_value), bins)

        P, edges = np.histogram(sample1, bins=bins, density=True)
        Q, _____ = np.histogram(sample2, bins=bins, density=True)

    dx = np.diff(edges)[0]
    x = edges[:-1] + dx / 2.
    ax.plot(x, P, label=sample_names[0], color=palette[0])
    if sample2 is not None:
        ax.plot(x, Q, label=sample_names[1], color=palette[1])
    ax.set_ylabel('p.d.f.')
    ax.set_xlabel(var_name)
    plt.legend()
    return ax
