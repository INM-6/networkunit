"""
The Baysian Estimation Effect Size  is introduced in Kruschke, J. (2012)
doi:10.1037/a0029146
"""
from __future__ import division
import numpy as np
import sciunit
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.ticker as mticker
import matplotlib.lines as mpllines
try:
    import best
    from pymc import Uniform, Normal, Exponential, NoncentralT, deterministic, Model, MCMC
    pymc = True
except:
    pymc = False


class best_effect_size(sciunit.Score):
    """
    Baysian Estimation Effect Size according to  Kruschke, J. (2012)

    Requires the test parameters:
        mcmc_iter : int (default 110000)
            Number of iterations of the Marcov-Chain-Monte-Carlo sampling.
        mcmc_burn : int (default 10000)
            Number of samples to be discarded to reduce potential
            correlations in the sampling sequence.
        effect_size_type : 'mode' (default), 'mean'
            How to determine an effect size value from the distribution
        assume_normal : bool
            If false, an additional 'normality' parameter is fitted to account
            for non-gaussianity of the data.
    """
    score = np.nan

    @classmethod
    def compute(self, observation, prediction,
                observation_name='observation',
                prediction_name='prediction',
                mcmc_iter=110000,
                mcmc_burn=10000,
                effect_size_type='mode',  # 'mean'
                assume_normal=False,
                **kwargs):
        if not pymc:
            raise ImportError('Module best or pymc could not be loaded!')

        data_dict = {observation_name: observation,
                     prediction_name: prediction}
        best_model = self.make_model(data_dict, assume_normal)
        M = MCMC(best_model)
        M.sample(iter=mcmc_iter, burn=mcmc_burn)

        group1_data = M.get_node(observation_name).value
        group2_data = M.get_node(prediction_name).value

        N1 = len(group1_data)
        N2 = len(group2_data)

        posterior_mean1 = M.trace('group1_mean')[:]
        posterior_mean2 = M.trace('group2_mean')[:]
        diff_means = posterior_mean1 - posterior_mean2

        posterior_std1 = M.trace('group1_std')[:]
        posterior_std2 = M.trace('group2_std')[:]

        pooled_var = ((N1 - 1) * posterior_std1 ** 2
                      + (N2 - 1) * posterior_std2 ** 2) / (N1 + N2 - 2)

        self.effect_size = diff_means / np.sqrt(pooled_var)

        stats = best.calculate_sample_statistics(self.effect_size)

        self.score = best_effect_size(stats[effect_size_type])
        self.score.mcmc_iter = mcmc_iter
        self.score.mcmc_burn = mcmc_burn
        self.score.data_size = [N1, N2]
        self.score.HDI = (stats['hdi_min'], stats['hdi_max'])
        self.HDI = self.score.HDI
        return self.score

    @classmethod
    def make_model(self, data, assume_normal=False):
        assert len(data) == 2, 'There must be exactly two data arrays'
        name1, name2 = sorted(data.keys())
        y1 = np.array(data[name1])
        y2 = np.array(data[name2])
        assert y1.ndim == 1
        assert y2.ndim == 1
        y = np.concatenate((y1, y2))

        mu_m = np.mean(y)
        mu_p = 0.000001 * 1 / np.std(y) ** 2

        sigma_low = np.std(y) / 1000
        sigma_high = np.std(y) * 1000

        # the five prior distributions for the parameters in our model
        group1_mean = Normal('group1_mean', mu_m, mu_p)
        group2_mean = Normal('group2_mean', mu_m, mu_p)
        group1_std = Uniform('group1_std', sigma_low, sigma_high)
        group2_std = Uniform('group2_std', sigma_low, sigma_high)
        nu_minus_one = Exponential('nu_minus_one', 1 / 29)

        if assume_normal:
            nu = 1000
        else:
            @deterministic(plot=False)
            def nu(n=nu_minus_one):
                out = n + 1
                return out

        @deterministic(plot=False)
        def lam1(s=group1_std):
            out = 1 / s ** 2
            return out

        @deterministic(plot=False)
        def lam2(s=group2_std):
            out = 1 / s ** 2
            return out

        group1 = NoncentralT(name1, group1_mean, lam1, nu, value=y1,
                             observed=True)
        group2 = NoncentralT(name2, group2_mean, lam2, nu, value=y2,
                             observed=True)
        return Model({'group1': group1,
                      'group2': group2,
                      'group1_mean': group1_mean,
                      'group2_mean': group2_mean,
                      'group1_std': group1_std,
                      'group2_std': group2_std,
                      })

    @classmethod
    def plot(self, sample1, sample2, ax=None, bins=20, barwidth=0.9,
             color='#6a9ced', **kwargs):
        if not np.nan_to_num(self.score):
            print('You must first compute the score by calling judge()')
            return None
        if ax is None:
            fig, ax = plt.subplots()

        hist, edges = np.histogram(self.effect_size, bins=bins, density=True)
        dx = np.diff(edges)[0]
        edges = edges[:-1]
        xvalues = edges + dx/2.
        ax.bar(xvalues, hist, width=barwidth*dx, color=color, edgecolor='w')
        ax.axvline(0, linestyle=':', color='k')
        hdi_line, = ax.plot(self.HDI, [0, 0],
                            lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(sum(self.HDI) / 2., 0.05, '95\% HDI',
                transform=trans,
                horizontalalignment='center',
                verticalalignment='bottom',
                )

        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')  # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])  # don't draw
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        ax.set_xlabel('Effect Size')
        return ax

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mBaysian Estimation Effect Size\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tIterations: {} \t Burn: {}" \
               .format(self.mcmc_iter, self.mcmc_burn) \
             + "\n\tEffect Size = {:.3f} \t HDI = ({:.3f}, {:.3f})\n\n" \
               .format(self.score, self.HDI[0], self.HDI[1])
