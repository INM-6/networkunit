import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import sciunit
from networkunit.scores import to_precision


class ks_distance(sciunit.Score):
    """
    Kolmogorov-Smirnov-Distance :math:`D_{KS}`

    .. math::
        D_\mathrm{KS} = \sup | \hat{P}(x) - \hat{Q}(x) |

    The KS-Distance measures the maximal vertical distance of the cumulative
    distributions :math:`\hat{P}` and :math:`\hat{Q}`. This measure is a sensitive tool for
    detecting differences in mean, variance and distribution type.

    The null hypothesis that the underlying distributions are identical is
    rejected when the :math:`D_{KS}` statistic is larger than a critical value, or
    equivalently when the corresponding p-value is less than the significance
    level.

    The computation is performed by the scipy.stats.ks_2samp() function.
    """
    score = np.nan

    @classmethod
    def compute(self, data_sample_1, data_sample_2, **kwargs):
        # Filter out nans and infs
        if len(np.shape(data_sample_1)) > 1:
            data_sample_1 = data_sample_1.flatten()
        if len(np.shape(data_sample_2)) > 1:
            data_sample_2 = data_sample_2.flatten()
        init_length = [len(smpl) for smpl in [data_sample_1, data_sample_2]]
        sample1 = np.array(data_sample_1)[np.isfinite(data_sample_1)]
        sample2 = np.array(data_sample_2)[np.isfinite(data_sample_2)]

        if init_length[0] - len(sample1) or init_length[1] - len(sample2):
            print("Warning: {} non-finite elements of the data samples were "
                  "filtered."
                  .format(sum(init_length)
                          - sum([len(s) for s in [sample1, sample2]])))

        DKS, pvalue = ks_2samp(sample1, sample2)
        score = ks_distance(DKS)
        score.distance = DKS
        score.pvalue = pvalue
        score.data_size = [len(sample1), len(sample2)]
        return score

    @classmethod
    def plot(self, sample1, sample2, ax=None, palette=None,
             include_scatterplot=False, var_name='Measured Parameter',
             sample_names=['observation', 'prediction'], **kwargs):
        if len(np.shape(sample1)) > 1:
            sample1 = sample1.flatten()
        if len(np.shape(sample2)) > 1:
            sample2 = sample2.flatten()
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_ylabel('CDF')
        ax.set_xlabel(var_name)
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        def alpha(color_inst, a):
            if type(color_inst) == str:
                if color_inst[0] == '#':
                    color_inst = colors.hex2color(color_inst)
            return [el + (1. - el) * (1-a) for el in color_inst]

        # plot cumulative distributions and scatterplot
        for i, sample in enumerate([sample1, sample2]):
            sorted_sample = np.sort(sample)
            sorted_sample = np.append(sorted_sample[0], sorted_sample)
            CDF = (np.arange(len(sample)+1)) / float(len(sample))
            ax.step(sorted_sample, CDF, where='post', color=palette[i],
                    label=sample_names[i])
            if include_scatterplot:
                ax.scatter(sorted_sample, [.99-i*.02]*len(sorted_sample),
                           color=palette[i], marker='D', linewidth=1)

        # calculate vertical distance
        N = len(sample1) + len(sample2)
        cdf_array = np.zeros((4, N))
        cdf_array[0] = np.append(sample1, sample2)
        cdf_array[1] = np.append(np.ones(len(sample1)),
                                 np.zeros(len(sample2)))
        cdf_array[2] = np.append(np.zeros(len(sample1)),
                                 np.ones(len(sample2)))
        sort_idx = np.argsort(cdf_array[0])
        cdf_array[0] = cdf_array[0][sort_idx]
        cdf_array[1] = np.cumsum(cdf_array[1][sort_idx]) / float(len(sample1))
        cdf_array[2] = np.cumsum(cdf_array[2][sort_idx]) / float(len(sample2))
        distance = cdf_array[1] - cdf_array[2]
        distance_plus = [d if d >= 0 else 0 for d in distance]
        distance_minus = [-d if d <= 0 else 0 for d in distance]

        # plot distance
        ax.plot(cdf_array[0], distance_plus, color=alpha(palette[0],.7), lw=.5)
        ax.fill_between(cdf_array[0], distance_plus, 0,
                        color=alpha(palette[0],.3))
        ax.plot(cdf_array[0], distance_minus, color=alpha(palette[1],.7), lw=.5)
        ax.fill_between(cdf_array[0], distance_minus, 0,
                        color=alpha(palette[1],.3))

        # plot max distance marker
        ax.axvline(cdf_array[0][np.argmax(abs(distance))],
                   color='.8', linestyle='--', linewidth=1.7)
        xlim_lower = min(min(sample1), min(sample2))
        xlim_upper = max(max(sample1), max(sample2))
        xlim_lower -= .03*(xlim_upper-xlim_lower)
        xlim_upper += .03*(xlim_upper-xlim_lower)
        ax.set_xlim(xlim_lower, xlim_upper)
        ax.set_ylim(0, 1)
        plt.legend()
        # plt.show()
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mKolmogorov-Smirnov-Distance\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tD_KS = {:.3f} \t p value = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))
