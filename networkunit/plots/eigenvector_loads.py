import numpy as np
from scipy.special import binom
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


def eigenvector_loads(EVs, color, hatch=None, ax=None,  ordered=False,
                          abs=False,  bin_size=.025, scaling=.85, alpha=0.001):
    # plots the loads of as many eigenvectors in EVs as there are colors

    if ax is None:
        fig, ax = plt.subplots()
    left, bottom, width, height = ax.get_position()._get_bounds()
    ax.set_position([left, bottom,
                     scaling * width, height])
    axhist = plt.axes([left + scaling * width, bottom,
                       (1-scaling) * width, height])

    if abs:
        EVs = np.absolute(EVs)
    if hatch is None:
        hatch = [''] * len(color)
    if ordered:
        vector_loads = np.sort(EVs.T[::-1], axis=-1)
        ax.set_xlabel('Neuron Rank')
        ax.invert_axis()
    else:
        vector_loads = EVs.T[::-1]
        ax.set_xlabel('Neuron')

    max_load = np.max(vector_loads[:len(color)])
    min_load = np.min(vector_loads[:len(color)])
    bin_num = (max_load - min_load) / bin_size
    edges = np.linspace(min_load, max_load, bin_num)
    for i, _ in enumerate(color):
        i = len(color) - i - 1
        ax.bar(np.arange(len(EVs.T[0]))+.5, vector_loads[i], 1., edgecolor='w',
               label=r'$v_{}$'.format(i+1), color=color[i], hatch=hatch[i])

        load_hist, _ = np.histogram(vector_loads[i], bins=edges, density=True)
        dx = edges[1] - edges[0]
        xvalues = edges[:-1] + dx/2.
        xvalues = np.append(np.append(xvalues[0]-dx, xvalues), xvalues[-1]+dx)
        load_hist = np.append(np.append(0, load_hist), 0)
        if hatch[i] == '//':
            ls = '--'
        else:
            ls = '-'
        axhist.plot(load_hist, xvalues, color=color[i], linestyle=ls)

        print("relevant neurons of vector {}:".format(i))
        print(load_significance(vector_loads[i], alpha=alpha))

    xvalues = np.linspace(-max_load, max_load, 100)
    normal_dist = mlab.normpdf(xvalues, 0, 1./np.sqrt(len(EVs)))
    axhist.plot(normal_dist, xvalues, color='k', linestyle=':', lw=2)

    sns.despine(ax=ax)
    axhist.axis('off')
    axhist.set_ylim(ax.get_ylim())
    ax.set_xlim(0, len(EVs.T[0])+1)
    ax.set_ylabel('Vector Load')
    handles, labels = ax.get_legend_handles_labels()
    plt.rcParams['legend.fontsize'] = plt.rcParams['axes.labelsize']+1
    ax.legend(handles[::-1],
              [r'$v_{}$'.format(j + 1) for j in range(len(color))],
              loc='best')
    return ax


def load_significance(ev, alpha=0.001):
    def twox_gaussian(x, mu, sig):
        return 2. * 1./(np.sqrt(2*np.pi)*sig) * \
               np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))
    def binom_p(j, th, N):
        p_th_inf, err = quad(twox_gaussian, th, np.inf, args=(0,1./np.sqrt(N)))
        p_0_th, err  = quad(twox_gaussian, 0, th, args=(0,1./np.sqrt(N)))
        return binom(N, j) * p_th_inf**j * p_0_th**(N-j)

    ev = np.absolute(ev)
    N = len(ev)
    sort_ids = []

    for l_count, neuron_id in enumerate(np.argsort(ev)[::-1]):
        if neuron_id not in sort_ids \
                and binom_p(l_count+1, ev[neuron_id], N) < alpha:
            sort_ids += [neuron_id]
        else:
            break

    return sort_ids