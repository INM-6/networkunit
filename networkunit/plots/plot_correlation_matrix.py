import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
try:
    import fastcluster
except:
    fastcluster_pkg = False
from scipy.special import binom
from scipy.spatial.distance import squareform
from copy import copy
from scipy.integrate import quad
import seaborn as sns

def correlation_matrix(matrix, ax=None, remove_autocorr=True, labels=None,
                        sort=False, cluster=False, linkmethod='ward',
                        dendrogram_args={}, sort_alpha=0.001,
                        vmin=None, vmax=None, cmap=None, center=0.,
                        robust=False, annot=None, fmt='.2g',
                        annot_kws=None, linewidths=0, linecolor='white',
                        cbar=True, cbar_kws=None, cbar_ax=None,
                        square=True, xticklabels='auto', limit_to_1=True,
                        yticklabels='auto', mask=None, addkwargs={},
                        rasterized=False,
                        **kwargs):

    pltmatrix = copy(matrix)
    order = np.arange(len(matrix))
    if sort:
        EWs, EVs = eigh(pltmatrix)
        order = reorder_matrix(EVs, EWs, alpha=sort_alpha)
        pltmatrix = pltmatrix[order, :][:, order]

    if limit_to_1:
        if np.max(pltmatrix) > 1:
            pltmatrix = pltmatrix / np.max(pltmatrix)

    if cluster:
        try:
            np.fill_diagonal(pltmatrix, 1)
            try:
                linkagematrix = linkage(squareform(1. - pltmatrix),
                                        method=linkmethod)
            except:
                if fastcluster_pkg:
                    print('using fastcluster')
                    linkagematrix = fastcluster.linkage(squareform(1. - pltmatrix),
                                                        method=linkmethod)
                else:
                    print('Clustering failed')
            dendro = dendrogram(linkagematrix, no_plot=True, **dendrogram_args)
            order = dendro['leaves']
            pltmatrix = pltmatrix[order, :][:, order]
        except Exception as e:
            print('Clustering failed')
            print(e)
            linkagematrix = None

    # if labels is None:
    #     labels = matrix.shape[0]/10
    #     if labels == 1:
    #         labels = 2
    # else:
    #     assert len(labels) == len(pltmatrix)

    if remove_autocorr:
        np.fill_diagonal(pltmatrix, 0)

    sns.heatmap(pltmatrix, ax=ax, vmin=vmin, vmax=vmax, square=square,
                            cbar=cbar, cmap=cmap, center=center,
                            robust=robust, annot=annot, fmt=fmt,
                            annot_kws=annot_kws, linewidths=linewidths,
                            linecolor=linecolor,
                            cbar_kws=cbar_kws, cbar_ax=cbar_ax,
                            xticklabels=xticklabels,
                            yticklabels=yticklabels, mask=mask,
                            rasterized=rasterized, **addkwargs)
    if sort or cluster:
        return order
    else:
        return np.arange(len(pltmatrix))


def reorder_matrix(EVs, EWs, alpha=0.001):
    def twox_gaussian(x, mu, sig):
        return 2. * 1./(np.sqrt(2*np.pi)*sig) * \
               np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))
    def binom_p(j, th, N):
        p_th_inf, err = quad(twox_gaussian, th, np.inf, args=(0,1./np.sqrt(N)))
        p_0_th, err  = quad(twox_gaussian, 0, th, args=(0,1./np.sqrt(N)))
        return binom(N, j) * p_th_inf**j * p_0_th**(N-j)

    N = len(EWs)
    EVs = np.absolute(EVs.T[::-1])
    sort_ids = []

    ew_sorting = np.argsort(EWs)[::-1]
    EWs = EWs[ew_sorting]
    EVs = EVs[ew_sorting]

    # significant vector loads for all eigenvectors
    for ev in EVs:
        for count, neuron_id in enumerate(np.argsort(ev)[::-1]):
            if neuron_id not in sort_ids \
                    and binom_p(count+1, ev[neuron_id], N) < alpha:
                sort_ids += [neuron_id]
            else:
                break
    # largest vector loads in the remaining neurons of all vectors
    for id in np.argsort(np.concatenate(EVs))[::-1]:
        neuron_id = id % N
        if neuron_id not in sort_ids:
            sort_ids += [neuron_id]
    return sort_ids