import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def sample_histogram(sample1, sample2=None, ax=None, bins=100, palette=None,
                     sample_names=['observation', 'prediction'], density=True,
                     var_name='Measured Parameter', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if palette is None:
        palette = [sns.color_palette()[0], sns.color_palette()[1]]
    if sample2 is None:
        P, edges1 = np.histogram(sample1, bins=bins, density=density)
    else:
        if isinstance(bins, int):
            max_value = max([max(sample1), max(sample2)])
            min_value = min([min(sample1), min(sample2)])
            bins = np.linspace(float(min_value), float(max_value), bins)
        P, edges1 = np.histogram(sample1, bins=bins, density=density)
        Q, edges2 = np.histogram(sample2, bins=bins, density=density)

    dx1 = np.diff(edges1)[0]
    x1 = edges1[:-1] + dx1 / 2.
    ax.plot(x1, P, label=sample_names[0], color=palette[0])

    if sample2 is not None:
        dx2 = np.diff(edges2)[0]
        x2 = edges2[:-1] + dx2 / 2.
        ax.plot(x2, Q, label=sample_names[1], color=palette[1])
    if density:
       ax.set_ylabel('p.d.f.')
    else:
        ax.set_ylabel('count')
    ax.set_xlabel(var_name)
    plt.legend()
    return ax