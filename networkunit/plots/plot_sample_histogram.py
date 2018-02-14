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
        ymax = max(P)
    else:
        max_value = max([max(sample1), max(sample2)])
        min_value = min([min(sample1), min(sample2)])
        edges = np.linspace(float(min_value), float(max_value), bins)

        P, edges = np.histogram(sample1, bins=edges, density=True)
        Q, _____ = np.histogram(sample2, bins=edges, density=True)

        ymax = max(max(P), max(Q))
        Q = np.append(np.append(0., Q), 0.)

    P = np.append(np.append(0., P), 0.)
    dx = np.diff(edges)[0]
    xvalues = edges[:-1] + dx / 2.
    xvalues = np.append(np.append(xvalues[0] - dx, xvalues),
                        xvalues[-1] + dx)
    ax.plot(xvalues, P, label=sample_names[0], color=palette[0])
    if sample2 is not None:
        ax.plot(xvalues, Q, label=sample_names[1], color=palette[1])
    ax.set_xlim(xvalues[0], xvalues[-1])
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Density')
    ax.set_xlabel(var_name)
    plt.legend()
    # plt.show()
    return ax