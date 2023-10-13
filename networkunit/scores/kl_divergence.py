import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import sciunit


class kl_divergence(sciunit.Score):
    r"""
    Kullback-Leibner Divergence :math:`D_{KL}(P||Q)`

    Calculates the difference of two sampled distributions P and Q in form of
    an entropy measure. The :math:`D_{KL}` measure is effectively the difference of the
    cross-entropy of the of both distribution P,Q and the entropy of P.
    :math:`D_{KL}` can be interpreted as the amount of information lost when
    approximating P by Q.

    .. math::
        D_\mathrm{KL}(P||Q) =\sum{i} P(i) \log_2 \frac{P(i)}{Q(i)}= H(P,Q) - H(P)

    The returned score is the symmetric version of the kl divergence

    .. math::
        D_\mathrm{KL}(P,Q) := \frac{1}{2} \left(D_\mathrm{KL}(P|Q) + D_\mathrm{KL}(Q|P)\right)

    Parameters:
    ----------
        kl_bin_size : float
            Bin size of the histogram, used to calculate the KL divergence.
    """
    score = np.nan

    @classmethod
    def compute(self, data_sample_1, data_sample_2, kl_bin_size=0.005, **kwargs):
        # filtering out nans
        sample1 = np.array(data_sample_1)[np.isfinite(data_sample_1)]
        sample2 = np.array(data_sample_2)[np.isfinite(data_sample_2)]

        max_value = max([max(sample1),max(sample2)])
        min_value = min([min(sample1),min(sample2)])
        bins = (max_value - min_value) / kl_bin_size
        bins = int(bins)
        edges = np.linspace(min_value, max_value, bins)

        P, edges = np.histogram(sample1, bins=edges, density=True)
        Q, _____ = np.histogram(sample2, bins=edges, density=True)
        # dx = np.diff(edges)[0]
        # edges = edges[:-1]
        # P *= dx
        # Q *= dx

        init_len = len(P)
        Qnot0 = np.where(Q != 0.)[0]
        P_non0 = P[Qnot0]
        Q_non0 = Q[Qnot0]
        Pnot0 = np.where(P_non0 != 0.)[0]
        Q_non0 = Q_non0[Pnot0]
        P_non0 = P_non0[Pnot0]
        final_len = len(P_non0)
        discard = init_len - final_len

        D_KL_PQ = entropy(P_non0, Q_non0, base=2)
        D_KL_QP = entropy(Q_non0, P_non0, base=2)

        D_KL = .5 * (D_KL_PQ + D_KL_QP)

        self.score = kl_divergence(D_KL)
        self.score.data_size = [len(sample1), len(sample2)]
        self.score.discarded_values = discard
        self.score.bins = len(edges)-1
        return self.score


    @classmethod
    def plot(self, data_sample_1, data_sample_2, ax=None, palette=None,
             var_name='Measured Parameter', kl_bin_size=0.005,
             sample_names=['observation', 'prediction'], **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_ylabel('Probability Density')
        ax.set_xlabel(var_name)
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        sample1 = np.array(data_sample_1)[np.isfinite(data_sample_1)]
        sample2 = np.array(data_sample_2)[np.isfinite(data_sample_2)]

        max_value = max([max(sample1),max(sample2)])
        min_value = min([min(sample1),min(sample2)])
        bins = (max_value - min_value) / kl_bin_size
        edges = np.linspace(min_value, max_value, bins)

        P, edges = np.histogram(sample1, bins=edges, density=True)
        Q, _____ = np.histogram(sample2, bins=edges, density=True)
        dx = np.diff(edges)[0]
        edges = edges[:-1]

        xvalues = edges + dx/2.
        xvalues = np.append(np.append(xvalues[0]-dx, xvalues), xvalues[-1]+dx)

        def secure_log(E, D):
            log = np.zeros_like(E)
            i = 0
            for e, d in zip(E, D):
                if e == 0 or d == 0:
                    log[i] = 0.
                else:
                    log[i] = np.log(e/d)
                i += 1
            return log

        diffy = .5 * (P - Q) * secure_log(P, Q.astype(float))
        P = np.append(np.append(0, P), 0)
        Q = np.append(np.append(0, Q), 0)
        filly = np.append(np.append(0., diffy), 0.)
        ax.fill_between(xvalues, filly, 0,  color='0.8', label='d/dx DKL')
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]
        ax.plot(xvalues, P, lw=2, color=palette[0], label=sample_names[0])
        ax.plot(xvalues, Q, lw=2, color=palette[1], label=sample_names[1])
        ax.set_xlim(xvalues[0], xvalues[-1])
        ax.set_yscale('log')
        plt.legend()
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mKullback-Leibler-Divergence\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tdiscarded: {}" \
               .format(self.discarded_values) \
             + "\n\tD_KL = {:.3f} \t bins = {}\n\n" \
               .format(self.score, self.bins)
