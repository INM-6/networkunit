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
        EWs1, EVs1 = eigh(matrix_1) # returns EWs in ascending order
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

        self.score = weighted_angle(np.pi/2. - weighted_mean_angle)
        self.score.data_size = (N, N)
        self.score.pvalue = -1  # ToDo
        return self.score

    @classmethod
    def weighted_angle_distribution(self, phi, N, B):
        # for weights ~ EW

        q = B / float(N)
        assert q >= 1
        x_min = (1 - np.sqrt(1. / q)) ** 2
        x_max = (1 + np.sqrt(1. / q)) ** 2

        def wigner_dist(x):
            y = q / (2 * np.pi) * np.sqrt((x_max - x) * (x - x_min)) / x
            if np.isnan(y):
                return 0
            else:
                return y

        y = lambda x: np.sin(x) ** (N - 2)
        angle_norm = quad(y, 0, np.pi)[0]

        def angle_dist(phi):
            if phi >= 0 and phi <= np.pi:
                return y(phi) / angle_norm
            else:
                return 0

        def combined_dist_integrand(x, z):
            return angle_dist(z / float(x)) * wigner_dist(x) * 1. / x

        def combined_dist(z):
            return quad(combined_dist_integrand, x_min, x_max, args=(z,))[0]

        if type(phi) == list or type(phi) == np.ndarray:
            return [combined_dist(z) for z in phi]
            # may has to be normalized
        else:
            return combined_dist(phi)

    @classmethod
    def plot(self, matrix_1, matrix_2, ax=None, B=None, palette=None, xlim=None,
             ylim=None, log=False, **kwargs):
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

        weights = np.sqrt((EWs1**2 + EWs2**2)/2.)
        weights = weights/sum(weights)
        weighted_mean_angle = np.average(angles, weights=weights)
        ax.axvline(weighted_mean_angle, color='k', ls='--', lw=3)

        edges = np.linspace(0, np.pi, 120)
        hist, _ = np.histogram(angles*weights*N, bins=edges, density=True)

        ax.bar(edges[:-1], hist, np.diff(edges)[0] * .99,
               color=palette[0], edgecolor='w',
               label=r'$\angle (\mathbf{v}_i,\mathbf{w}_j)$'
                     + r'$: i,j\in [1,{}]$'.format(str(N)))

        # analytical distribution for random activity
        if B is not None:
            x = np.linspace(0, np.pi, 120)
            y = self.weighted_angle_distribution(x, N, B)
            norm = np.sum(y) * (x[1] - x[0])
            ax.plot(x, y / norm, color=palette[1])

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
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mEigenvector Angle Score\033[0m" \
             + "\n\tdatasize: {} x {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tscore = {:.3f} \t pvalue = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))