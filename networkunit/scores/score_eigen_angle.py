import numpy as np
from scipy.linalg import eigh
from scipy.special import binom
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import sciunit
from networkunit.scores import to_precision


class eigen_angle(sciunit.Score):
    """

    """
    score = np.nan

    @classmethod
    def compute(self, matrix_1, matrix_2, alpha, all_to_all, two_sided, **kwargs):
        EWs1, EVs1 = eigh(matrix_1)
        EWs2, EVs2 = eigh(matrix_2)
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]

        N = len(matrix_1)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            if all_to_all:
                angles = np.arccos(M.flatten())
                NB = N**2
            else:
                angles = np.arccos(np.diag(M))
                NB = N

        angles[np.where(np.isnan(angles))[0]] = -0.

        def angle_dist(x):
            return np.power(np.sin(x), N-2)

        def binom_p(j, th, NB):
            p_0_th, err = quad(angle_dist, 0, th)
            p_th_pi, err = quad(angle_dist, th, np.pi)
            norm = 1./quad(angle_dist, 0, np.pi)[0]
            return binom(NB, j) * norm**NB * p_0_th**j * p_th_pi**(NB - j)

        j_max = 0
        th = 0

        for count, phi in enumerate(np.sort(angles)):
            if binom_p(count+1, phi, NB) > alpha:
                j_max = count
                th = phi
                break

        # ToDo: Calculate pvalue

        self.data_size = (N,N)
        self.alpha = alpha
        self.all_to_all = all_to_all
        self.two_sided = two_sided
        self.threshold = th
        self.pvalue = -1 # ToDo
        self.score = eigen_angle(j_max/float(N))
        return self.score


    @classmethod
    def plot(self, matrix_1, matrix_2, all_to_all, two_sided, alpha,
             ax=None, palette=None, xlim=None,
             ylim=None, log=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_ylabel('Angle Density')
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        EWs1, EVs1 = eigh(matrix_1)
        EWs2, EVs2 = eigh(matrix_2)
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]

        N = len(matrix_1)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            if self.all_to_all:
                angles = np.arccos(M.flatten())
                NB = N**2
            else:
                angles = np.arccos(np.diag(M))
                NB = N

        angles[np.where(np.isnan(angles))[0]] = -0.

        res = 10000
        rand_angles = np.linspace(0, np.pi, res)
        dist = np.array([np.sin(a) ** (N - 2) for a in rand_angles])
        dist = dist / (np.sum(dist) * np.pi / res)

        ax.plot(rand_angles, dist, color='0.3', lw=2, ls='--',
                label=r'$\sin^{}\phi$'.format('{' + str(N - 2) + '}'))

        edges = np.linspace(0, np.pi, 120)
        hist, _ = np.histogram(angles, bins=edges, density=True)

        ax.bar(edges[:-1], hist, np.diff(edges)[0] * .99,
               color=sns.color_palette()[0], edgecolor='w',
               label=r'$\angle (\mathbf{v}_i,\mathbf{w}_j)$'
                     + r'$: i,j\in [1,{}]$'.format(str(N)))

        score = self.compute(matrix_1, matrix_2, alpha, all_to_all, two_sided)
        ax.axvline(score.threshold, color='r', ls=':')

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
        log = False
        if log:
            ax.set_yscale('log')
        plt.legend()
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mEigen Angle Score\033[0m" \
             + "\n\tdatasize: {} x {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n" + "\ttwo sided" if self.two_sided else "" \
             + "\t all to all angles" if self.all_to_all else "" \
             + "\n\tscore = {:.3f} \t pvalue = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))