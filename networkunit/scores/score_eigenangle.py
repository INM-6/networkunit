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

    Parameters
    ----------
        all_to_all : bool (default False)
            If False, only the angle between first and first eigenvectors is
            calculated, second and second, and so on, else all combinations are
            taken into account.

    """
    score = np.nan

    @classmethod
    def compute(self, matrix_1, matrix_2, all_to_all=False, bin_num=None,
                binsize=None, t_start=None, t_stop=None, **kwargs):
        if bin_num is None:
            if binsize is not None and (t_start is not None and t_stop is not None):
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
        for count, ev in enumerate(EVs1):
            EVs1[count] = ev * np.sign(ev[np.argmax(np.absolute(ev))])
        for count, ev in enumerate(EVs2):
            EVs2[count] = ev * np.sign(ev[np.argmax(np.absolute(ev))])

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
            angle_nbr = 1
            weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
        else:
            if all_to_all:
                angles = np.arccos(M.flatten())
                weights = np.zeros((len(EWs1), len(EWs1)))
                for count1, ew1 in enumerate(EWs1):
                    for count2, ew2 in enumerate(EWs2):
                        weights[count1, count2] = np.sqrt(
                            (ew1 ** 2 + ew2 ** 2) / 2.)
                weights = weights.flatten()
                angle_nbr = N ** 2
            else:
                angles = np.arccos(np.diag(M))
                weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
                angle_nbr = N

        weights = weights / sum(weights)
        sample_angles = angles * weights * angle_nbr
        weighted_mean_angle = np.mean(sample_angles)

        x = np.linspace(0, np.pi, 120)
        y = self.weighted_angle_distribution(x, N, B=bin_num, use_gaus=all_to_all)
        norm = np.sum(y) * (x[1] - x[0])
        y = y / norm
        sample_std = np.std(self.inverse_transform_sampling(x, y, n_samples=10**6))
        sigma = sample_std / (2. * np.sqrt(angle_nbr))
        pvalue = quad(mlab.normpdf, 0., weighted_mean_angle, args=(np.pi/2., sigma))[0]

        weighted_angle_score = (np.pi/2. - weighted_mean_angle) / (np.pi/2.)

        self.score = eigenangle(weighted_angle_score)
        self.score.data_size = (N, N)
        self.score.pvalue = pvalue
        return self.score

    @classmethod
    def weighted_angle_distribution(self, phi, N, B, use_gaus=False):
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

        def gaussian(x, mu, sig):
            return 1./(sig*np.sqrt(2.*np.pi)) * \
                   np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        def eff_dist(x):
            if use_gaus:
                return gaussian(x,1.,np.sqrt(N)/100.)
            else:
                return wigner_dist(x)

        y = lambda x: np.sin(x) ** (N - 2)
        angle_norm = quad(y, 0, np.pi)[0]

        def angle_dist(phi):
            if phi >= 0 and phi <= np.pi:
                return y(phi) / angle_norm
            else:
                return 0

        def combined_dist_integrand(x, z):
            return angle_dist(z / float(x)) * eff_dist(x) * 1. / x

        def combined_dist(z):
            return quad(combined_dist_integrand, x_min, x_max, args=(z,))[0]

        if type(phi) == list or type(phi) == np.ndarray:
            return [combined_dist(z) for z in phi]
            # may has to be normalized
        else:
            return combined_dist(phi)

    @classmethod
    def inverse_transform_sampling(self, x, y, n_samples=10**6):
        x = x + np.diff(x)[0] / 2.
        x = np.append(np.array([0]), x)
        cum_values = np.zeros(len(x))
        cum_values[1:] = np.cumsum(y * np.diff(x))
        inv_cdf = interpolate.interp1d(cum_values, x)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    @classmethod
    def plot(self, matrix_1, matrix_2, ax=None, bin_num=None, palette=None,
             binsize=None, t_start=None, t_stop=None,
             log=False, all_to_all=False, **kwargs):

        if bin_num is None:
            if binsize is not None and (t_start is not None and t_stop is not None):
                    bin_num = float((t_stop - t_start) / binsize)
            else:
                raise ValueError('To few parameters to compute bin_num!')

        if ax is None:
            fig, ax = plt.subplots()
        if palette is None:
            palette = [sns.color_palette('Set2')[1], sns.color_palette('Set2')[4]]

        N = len(matrix_1)
        EWs1, EVs1 = eigh(matrix_1)  # returns EWs in ascending order
        EWs2, EVs2 = eigh(matrix_2)
        EWs1 = EWs1[::-1]
        EWs2 = EWs2[::-1]
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]
        for count, ev in enumerate(EVs1):
            EVs1[count] = ev * np.sign(ev[np.argmax(np.absolute(ev))])
        for count, ev in enumerate(EVs2):
            EVs2[count] = ev * np.sign(ev[np.argmax(np.absolute(ev))])

        M = np.dot(EVs1, EVs2.T)
        M[np.argwhere(M > 1)] = 1.

        if len(M) == 1:
            angles = np.arccos(M[0])
            angle_nbr = 1
            weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
        else:
            if all_to_all:
                angles = np.arccos(M.flatten())
                weights = np.zeros((len(EWs1), len(EWs1)))
                for count1, ew1 in enumerate(EWs1):
                    for count2, ew2 in enumerate(EWs2):
                        weights[count1, count2] = np.sqrt(
                            (ew1 ** 2 + ew2 ** 2) / 2.)
                weights = weights.flatten()
                angle_nbr = N ** 2
            else:
                angles = np.arccos(np.diag(M))
                weights = np.sqrt((EWs1 ** 2 + EWs2 ** 2) / 2.)
                angle_nbr = N

        weights = weights / sum(weights)
        sample_angles = angles * weights * angle_nbr

        ax.set_xticks(np.array([0, 0.125, .25, .375, .5, .625, .75, .875, 1]) * np.pi)
        ax.set_xticklabels(['0', r'$\frac{1}{8}\pi$', r'$\frac{1}{4}\pi$',
                            r'$\frac{3}{8}\pi$', r'$\frac{1}{2}\pi$',
                            r'$\frac{5}{8}\pi$', r'$\frac{3}{4}\pi$',
                            r'$\frac{7}{8}\pi$', r'$\pi$'], fontsize=18)

        ax.set_xlabel(r'Weighted Angle$')

        edges = np.linspace(0, np.pi, 120)
        hist, _ = np.histogram(sample_angles, bins=edges, density=True)

        if all_to_all:
            label = [r'$w_{ij}\cdot\angle (\mathbf{v}_i,\mathbf{w}_j):$',
                     '$\mathbf{v}_i,\mathbf{w}_j \in $'+'$R^{}$'.format('{'+str(N)+'}')]
        else:
            label = [r'$w_{ii}\cdot\angle (\mathbf{v}_i,\mathbf{w}_i):$',
                     '$\mathbf{v}_i,\mathbf{w}_i \in $'+'$R^{}$'.format('{'+str(N)+'}')]
        ax.bar(edges[:-1], hist, np.diff(edges)[0] * .99,
               color=palette[1], edgecolor='w',
               label='{}'.format(label[0]))
        ax.bar([0], [0], width=0, color='w', edgecolor='w',
               label='{}'.format(label[1]))

        x = np.linspace(0, np.pi, 120)
        y = self.weighted_angle_distribution(x, N, B=bin_num, use_gaus=all_to_all)
        norm = np.sum(y) * (x[1] - x[0])
        ax.plot(x, np.array(y) / norm, color=palette[0],
                label='RMT Prediction')
        ax.axvline(np.mean(sample_angles), color='k', ls='--', label='weighted mean angle')

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