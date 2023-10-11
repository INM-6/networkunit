import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import sciunit
from networkunit.scores import to_precision


class mwu_statistic(sciunit.Score):
    r"""
        Mann-Whitney-U test

    .. math::
        U_i = R_i - \frac{n_i(n_i + 1)}{2}\\
        U = min(U_1,U_2)

    With the rank sum R and the sample size :math:`n_i`.

    The Mann-Whitney U is a rank statistic which test the null hypothesis
    that a random value of sample 1 is equally likely to be larger or a smaller
    value than a randomly chosen value of sample 2.

    The U_i statistic is in the range of [0,n_1 n_2],
    and the U=min(U_1,U_2) statistic is in the range of [0,n_1*n_2/2].

    For sample sizes >20, U follows approximately a normal distribution.
    With this assumption a p-value can be inferred. The null hypothesis is
    consequently rejected when the p-value is less than the significance level.
    """
    score = np.nan

    @classmethod
    def compute(self, sample1, sample2, bins=100, excl_nan=True,
                **kwargs):
        # filtering out nans
        init_len = np.array([len(sample1), len(sample2)])
        if excl_nan:
            sample1 = np.array(sample1)[np.isfinite(sample1)]
            sample2 = np.array(sample2)[np.isfinite(sample2)]
        final_len = np.array([len(sample1), len(sample2)])
        discard = sum(init_len-final_len)

        if len(sample1) < 20 or len(sample2) < 20:
            raise Warning('The sample size is too small. '
                          'The test might lose its validity!')

        U, pvalue = mannwhitneyu(sample1, sample2, alternative=None)
        pvalue *= 2

        self.score = mwu_statistic(U)
        self.score.data_size = [len(sample1), len(sample2)]
        self.score.discarded_values = discard
        self.score.pvalue = pvalue
        return self.score

    @classmethod
    def plot(self, sample1, sample2, ax=None, palette=None,
             var_name='', excl_nan=True,
             sample_names=['observation', 'prediction'], **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_ylabel('Rank')
        ax.set_xlabel(var_name)
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        if excl_nan:
            sample1 = np.array(sample1)[np.isfinite(sample1)]
            sample2 = np.array(sample2)[np.isfinite(sample2)]

        N = len(sample1) + len(sample2)
        ranks = [[0] * 2, [0] * N]
        ranks[0][:len(sample1)] = sample1
        ranks[1][:len(sample1)] = [sample_names[0]] * len(sample1)
        ranks[0][len(sample1):] = sample2
        ranks[1][len(sample1):] = [sample_names[1]] * len(sample2)
        ranks[0] = rankdata(ranks[0])

        dataframe = DataFrame({'Rank': ranks[0],
                               'Group': ranks[1],
                               'Rank Density Estimate': np.zeros(N)})

        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        sns.violinplot(data=dataframe, x='Rank Density Estimate',
                       y='Rank',
                       hue='Group', split=True, palette=palette,
                       inner='quartile', cut=0, ax=ax,
                       scale_hue=True, scale='width')
        ax.set_ylabel(var_name + ' Rank')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.set_ylim(0, len(sample1)+len(sample2))
        plt.legend()
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\033[4mMann-Whitney-U-Test\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tU = {:.3f}   \t p value = {}" \
               .format(self.score, to_precision(self.pvalue, 2))
