import numpy as np
from scipy.linalg import eigh
from scipy.special import binom
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import sciunit
from networkunit.scores import to_precision


class weighted_angle(sciunit.Score):
    """
    Eigenvectors can represent dominant features of a network. Similar networks
    should therefore have similar eigenvectors such that the angle between them
    should be small. The angles are counted which are
    significantly smaller (with respect to significance level alpha) than
    expected from random network activity of N neurons. The score is then the
    number of small angles / N.
    """
    score = np.nan

    @classmethod
    def compute(self, matrix_1, matrix_2, **kwargs):
        EWs1, EVs1 = eigh(matrix_1) # retruns EWs in ascending order
        EWs2, EVs2 = eigh(matrix_2)
        EWs1 = EWs1[::-1]
        EWs2 = EWs2[::-1]
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]

        N = len(matrix_1)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            angles = np.arccos(np.diag(M))

        weights = EWs1 * EWs2
        weights = weights/sum(weights)
        weighted_mean_angle = np.average(angles, weights=weights)

        # ToDo: Calculate pvalue

        self.score = weighted_angle(weighted_mean_angle)
        self.score.data_size = (N, N)
        self.score.pvalue = -1  # ToDo
        return self.score


    @classmethod
    def plot(self, matrix_1, matrix_2, ax=None, palette=None, xlim=None,
             ylim=None, log=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_ylabel('Angle Density')
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        EWs1, EVs1 = eigh(matrix_1)
        EWs2, EVs2 = eigh(matrix_2)
        EWs1 = EWs1[::-1]
        EWs2 = EWs2[::-1]
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]

        N = len(matrix_1)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            angles = np.arccos(np.diag(M))

        res = 10000
        rand_angles = np.linspace(0, np.pi, res)
        dist = np.array([np.sin(a) ** (N - 2) for a in rand_angles])
        dist = dist / (np.sum(dist) * np.pi / res)

        # ax.plot(rand_angles, dist, color='0.3', lw=2, ls='--',
        #         label=r'$\sin^{}\phi$'.format('{' + str(N - 2) + '}'))

        weights = np.sqrt((EWs1**2 + EWs2**2)/2.)
        weights = weights/sum(weights)
        weighted_mean_angle = np.average(angles, weights=weights)
        ax.axvline(weighted_mean_angle, color='r', ls=':', lw=2)

        edges = np.linspace(0, np.pi, 120)
        hist, _ = np.histogram(angles*weights*N, bins=edges, density=True)

        print np.mean(angles*weights*N)
        print weighted_mean_angle

        ax.bar(edges[:-1], hist, np.diff(edges)[0] * .99,
               color=palette[0], edgecolor='w',
               label=r'$\angle (\mathbf{v}_i,\mathbf{w}_j)$'
                     + r'$: i,j\in [1,{}]$'.format(str(N)))

        ax.set_xticks(
            np.array([0, 0.125, .25, .375, .5, .625, .75, .875, 1]) * np.pi)
        ax.set_xticklabels(['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$',
                            r'$\frac{3}{8}\pi$', r'$\frac{\pi}{2}$',
                            r'$\frac{5}{8}\pi$', r'$\frac{3}{4}\pi$',
                            r'$\frac{7}{8}\pi$', r'$\pi$'])
        ax.set_xlabel(r'Plane Angle in $\mathtt{R}$'
                      + r'$^{}$'.format('{' + str(N) + '}'))
        sns.despine()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if log:
            ax.set_yscale('log')
        plt.legend()
        return ax, angles*weights*N

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mEigenvector Angle Score\033[0m" \
             + "\n\tdatasize: {} x {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tscore = {:.3f} \t pvalue = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))