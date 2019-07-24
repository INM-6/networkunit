import numpy as np
from scipy.linalg import eigh
from scipy.special import binom
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import sciunit
from networkunit.scores import to_precision
import matplotlib.mlab as mlab
from scipy.integrate import quad
import scipy.interpolate as interpolate


class eigenangle(sciunit.Score):
    """
    The eigenangle score evaluates whether two correlation matrices have
    similar non-random elements by calculating the significance of the angles
    between the corresponding eigenvectors.
    Either the binsize or the number of bins must be provides to perform the
    signficnace test.
    """
    score = np.nan

    @classmethod
    def compute(self, matrix_1, matrix_2, bin_num=None,
                binsize=None, t_start=None, t_stop=None, **kwargs):
        if bin_num is None:
            if binsize is not None \
            and (t_start is not None and t_stop is not None):
                    bin_num = float((t_stop - t_start) / binsize)
            else:
                raise ValueError('To few parameters to compute bin_num!')
        N = len(matrix_1)
        EWs1, EVs1 = eigh(matrix_1) # returns EWs in ascending order
        EWs2, EVs2 = eigh(matrix_2)
        EWs1 = EWs1[::-1]
        EWs2 = EWs2[::-1]
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]
        for count, (ev1, ev2) in enumerate(zip(EVs1, EVs2)):
            EVs1[count] = ev1 * np.sign(ev1[np.argmax(np.absolute(ev1))])
            EVs2[count] = ev2 * np.sign(ev2[np.argmax(np.absolute(ev2))])
            EVs1[count] /= np.linalg.norm(ev1)
            EVs2[count] /= np.linalg.norm(ev2)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            angles = np.arccos(np.diag(M))

        weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
        smallness = 1 - angles / (np.pi/2.)
        weights = weights / sum(weights) * N
        weighted_smallness = smallness * weights
        similarity_score = np.mean(weighted_smallness)

        pvalue = quad(self.null_distribution,
                      similarity_score, np.inf,
                      args=(N, bin_num))[0]

        self.score = eigenangle(similarity_score)
        self.score.data_size = (N, N)
        self.score.pvalue = pvalue
        return self.score

    @classmethod
    def null_distribution(self, eta, N, B, return_plotting_dist=False):
        # for weights ~ EW

        q = B / float(N)
        assert q >= 1

        def marchenko_pastur(x, alpha):
            assert alpha >= 1
            x_min = (1 - np.sqrt(1. / alpha)) ** 2
            x_max = (1 + np.sqrt(1. / alpha)) ** 2
            y = alpha / (2 * np.pi * x) * np.sqrt((x_max - x) * (x - x_min))
            if np.isnan(y):
                return 0
            else:
                return y

        def weight_dist(x):
            # ToDo: add alternative distributions for e.g. asymmetric matrices
            return merchenko_pastur(x)

        def angle_smallness_dist(D, N):
            if D >= -1 and D <= 1:
                return math.gamma(N/2.) / (np.sqrt(np.pi) \
                     * math.gamma((N-1)/2)) \
                     * np.pi/2 * np.cos(D*np.pi/2)**(N-2)
            else:
                return 0

        def weighted_smallness_dist(D, N, alpha):
            x_min = (1 - np.sqrt(1. / alpha)) ** 2
            x_max = (1 + np.sqrt(1. / alpha)) ** 2

            integrand = lambda x, _D, _N, _alpha: \
                               angle_smallness_dist(_D / float(x), _N) \
                               * weight_dist(x, _alpha) * 1. / x
            return sc.integrate.quad(integrand, x_min, x_max,
                                     args=(D,N,alpha,))[0]

        def similarity_score_distribution(eta, N, alpha):
            integrand = lambda x, N_, alpha_: \
                               x**2 * weighted_smallness_dist(x, N_, alpha_)
            var = sc.integrate.quad(integrand,
                                    -np.infty, np.infty,
                                    args=(N,alpha,))[0]
            sigma = np.sqrt(var/N)
            return sc.stats.norm.pdf(eta, 0, sigma)

        if return_plotting_dist:
            return weighted_smallness_dist
        else:
            return similarity_score_distribution(eta, N, q)


    @classmethod
    def plot(self, matrix_1, matrix_2, ax=None, bin_num=None, palette=None,
             binsize=None, t_start=None, t_stop=None, log=False, **kwargs):

        if bin_num is None:
            if binsize is not None \
            and (t_start is not None and t_stop is not None):
                    bin_num = float((t_stop - t_start) / binsize)
            else:
                raise ValueError('To few parameters to compute bin_num!')
        N = len(matrix_1)
        EWs1, EVs1 = eigh(matrix_1) # returns EWs in ascending order
        EWs2, EVs2 = eigh(matrix_2)
        EWs1 = EWs1[::-1]
        EWs2 = EWs2[::-1]
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]
        for count, (ev1, ev2) in enumerate(zip(EVs1, EVs2)):
            EVs1[count] = ev1 * np.sign(ev1[np.argmax(np.absolute(ev1))])
            EVs2[count] = ev2 * np.sign(ev2[np.argmax(np.absolute(ev2))])
            EVs1[count] /= np.linalg.norm(ev1)
            EVs2[count] /= np.linalg.norm(ev2)

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
        else:
            angles = np.arccos(np.diag(M))

        weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
        smallness = 1 - angles / (np.pi/2.)
        weights = weights / sum(weights) * N
        weighted_smallness = smallness * weights
        similarity_score = np.mean(weighted_smallness)
        
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_xlabel(r'Weighted Angle-Smallness$')
        edges = np.linspace(0, 1, 120)
        hist, _ = np.histogram(weighted_smallness, bins=edges, density=True)

        ax.bar(edges[:-1], hist, np.diff(edges)[0] * .99,
               color=palette[1], edgecolor='w')

        weighted_smallness_dist = self.null_distribution(eta=0, N=N, B=bin_num,
                                                    return_plotting_dist=True)

        y = [weighted_smallness_dist(x, N=N, alpha=bin_num/N) for x in edges]
        norm = np.sum(y) * (edges[1] - edges[0])
        ax.plot(x, np.array(y) / norm, color=palette[0],
                label='Prediction')
        ax.axvline(np.mean(weighted_smallness), color='k', ls='--',
                   label='Samples')

        ax.set_yticks([])
        plt.legend()
        sns.despine(left=True)
        if log:
            ax.set_yscale('log')
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mEigenangle Score\033[0m" \
             + "\n\tdatasize: {} x {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tscore = {:.3f} \t pvalue = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))
