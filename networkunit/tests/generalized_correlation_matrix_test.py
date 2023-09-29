from networkunit.tests.correlation_matrix_test import correlation_matrix_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots import alpha as _alpha
from elephant.spike_train_correlation import cch
import matplotlib.pyplot as plt
# from matplotlib import colors, colorbar
import seaborn as sns
from copy import copy

from quantities import ms, Quantity
import numpy as np


class generalized_correlation_matrix_test(correlation_matrix_test):
    """
    Test to compare the different kinds of correlation matrices of a set of
    spiking neurons in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.

    Parameters:
    ----------
        bin_size: quantity, None (default: 2*ms)
        Size of bins used to calculate the correlation coefficients.
    num_bins: int, None (default: None)
        Number of bins within t_start and t_stop used to calculate
        the correlation coefficients.
    t_start: quantity, None
        Start of time window used to calculate the correlation coefficients.
    t_stop: quantity, None
        Stop of time window used to calculate the correlation coefficients.
    nan_to_num: bool
        If true, np.nan are set to 0, and np.inf to largest finite float.
    binary: bool
        If true, the binned spike trains are set to be binary.
    cluster_matrix : bool
        If true, the matrix is clustered by the hierarchical cluster algorithm
        scipy.cluster.hierachy.linkage() with 'method' determined by the
        cluster_method.
    cluster_method : string (default: 'ward')
        Method for the hierarchical clustering if cluster_matrix=True
    remove_autocorr: bool
        If true, the diagonal values of the matrix are set to 0.
    edge_threshold: float
        Passed to draw_graph() and determines the threshold above which edges
        are draw in the graph corresponding to the matrix.
    maxlag : int
        Maximum shift (in number of bins) between spike trains which should
        still be considered in the calculating the correlation measure.
    time_reduction: 'sum', 'max', 'threshold x.x'
        Method how to include lagged correlations between spike trains.
        sum - calculates the sum of the normalized CCH within +- maxlag
        max - takes the maximum of the CCH within +- maxlag
        threshold x.x - sums up the part of the CCH above the threshold x.x
                        and within +- maxlag
    """

    required_capabilities = (ProducesSpikeTrains, )

    default_params = {**correlation_matrix_test.default_params,
                      'maxlag': 100,  # in bins
                      'time_reduction': 'threshold 0.13',
                      'binary': False
                      }

    def generate_cc_matrix(self, spiketrains=None, model=None):

        if hasattr(model, 'cch_array')\
             and 'bin_size{}_maxlag{}'.format(self.params['bin_size'],
                                              self.params['maxlag'])\
             in model.cch_array:
            cch_array = model.cch_array['bin_size{}_maxlag{}'\
                .format(self.params['bin_size'], self.params['maxlag'])]
        else:
            cch_array = self.generate_cch_array(spiketrains=spiketrains,
                                                **self.params)
            if model is not None:
                if not hasattr(model, 'cch_array'):
                    model.cch_array = {}
                model.cch_array['bin_size{}_maxlag{}'\
                        .format(self.params['bin_size'], self.params['maxlag'])] = cch_array

        pairs_idx = np.triu_indices(len(spiketrains), 1)
        pairs = [[i, j] for i, j in zip(pairs_idx[0], pairs_idx[1])]
        if 'time_reduction' not in self.params:
            raise KeyError("A method for 'time_reduction' needs to be set!")
        return self.generalized_cc_matrix(cch_array, pairs,
                                          self.params['time_reduction'])

    def generalized_cc_matrix(self, cch_array, pair_ids, time_reduction,
                              rescale=False, **kwargs):
        B = len(np.squeeze(cch_array)[0])
        if time_reduction == 'sum':
            cc_array = np.sum(np.squeeze(cch_array), axis=1)
            if rescale:
                cc_array = cc_array / float(B)
        if time_reduction == 'max':
            cc_array = np.amax(np.squeeze(cch_array), axis=1)
        if time_reduction[:3] == 'lag':
            lag = int(time_reduction[3:])
            cc_array = np.squeeze(cch_array)[:, B/2 + lag]
        if time_reduction[:9] == 'threshold':
            th = float(time_reduction[10:])
            th_cch_array = np.array([a[a > th] for a in np.squeeze(cch_array)])
            if rescale:
                cc_array = np.array([np.sum(cch)/float(len(cch)) if len(cch)
                                     else np.sum(cch)
                                     for cch in th_cch_array])
            else:
                cc_array = np.array([np.sum(cch) for cch in th_cch_array])
        N = len(cc_array)
        dim = .5*(1 + np.sqrt(8.*N + 1))
        assert not dim - int(dim)
        dim = int(dim)
        cc_mat = np.ones((dim, dim))
        for count, (i, j) in enumerate(pair_ids):
            cc_mat[i, j] = cc_array[count]
            cc_mat[j, i] = cc_array[count]
        return cc_mat

    def generate_cch_array(self, spiketrains, maxlag=None, model=None,
                           **kwargs):
        if 'bin_size' in self.params:
            bin_size = self.params['bin_size']
        elif 'num_bins' in self.params:
            t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
            tmin = min(t_lims, key=lambda f: f[0])[0]
            tmax = max(t_lims, key=lambda f: f[1])[1]
            T = tmax - tmin
            bin_size = T / float(self.params['num_bins'])
        else:
            raise AttributeError("Neither bin size or number of bins was set!")
        if maxlag is None:
            maxlag = self.params['maxlag']
        else:
            self.params['maxlag'] = maxlag
        if type(maxlag) == Quantity:
            maxlag = int(float(maxlag.rescale('ms'))
                         / float(bin_size.rescale('ms')))

        if hasattr(model, 'cch_array') and \
           'bin_size{}_maxlag{}'.format(bin_size, maxlag) in model.cch_array:
            cch_array = model.cch_array['bin_size{}_maxlag{}'.format(bin_size,
                                                                    maxlag)]
        else:
            try:
                from mpi4py import MPI
                mpi = True
            except:
                mpi = False
            N = len(spiketrains)
            B = 2 * maxlag + 1
            pairs_idx = np.triu_indices(N, 1)
            pairs = [[i, j] for i, j in zip(pairs_idx[0], pairs_idx[1])]
            if mpi:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                Nnodes = comm.Get_size()
                print('Using MPI with {} node'.format(Nnodes))
                comm.Barrier()
                if rank == 0:
                    split = np.array_split(pairs, Nnodes)
                else:
                    split = None
                pair_per_node = int(np.ceil(float(len(pairs)) / Nnodes))
                split_pairs = comm.scatter(split, root=0)
            else:
                split_pairs = pairs
                pair_per_node = len(pairs)

            cch_array = np.zeros((pair_per_node, B))
            max_cc = 0
            for count, (i, j) in enumerate(split_pairs):
                binned_sts_i = self.robust_BinnedSpikeTrain(spiketrains[i],
                                                            bin_size=bin_size)
                binned_sts_j = self.robust_BinnedSpikeTrain(spiketrains[j],
                                                            bin_size=bin_size)
                cch_array[count] = np.squeeze(cch(binned_sts_i,
                                                  binned_sts_j,
                                                  window=[-maxlag, maxlag],
                                                  cross_corr_coef=True,
                                                  )[0])
                max_cc = max([max_cc, max(cch_array[count])])
            if mpi:
                pop_cch = comm.gather(cch_array, root=0)
                pop_max_cc = comm.gather(max_cc, root=0)
                if rank == 0:
                    cch_array = pop_cch
                    max_cc = pop_max_cc

        if model is not None:
            if not hasattr(model, 'cch_array'):
                model.cch_array = {}
            model.cch_array['bin_size{}_maxlag{}'.format(bin_size, maxlag)] = cch_array
        return cch_array

    def _calc_color_array(self, cch_array, threshold=.5, **kwargs):
        # save as array with only >threshold value
        # and a second array with their (i,j,t)
        squeezed_cch_array = np.squeeze(cch_array)
        N = len(squeezed_cch_array)
        B = len(squeezed_cch_array[0])
        dim = int(.5 * (1 + np.sqrt(8. * N + 1)))
        pairs = np.triu_indices(dim, 1)
        pair_ids = [[i, j] for i, j in zip(pairs[0], pairs[1])]
        binnums = np.arange(B)

        try:
            from mpi4py import MPI
            mpi = True
        except:
            mpi = False
        if mpi:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            Nnodes = comm.Get_size()
            print('rank', rank, 'and size (Nnodes)', Nnodes)
            comm.Barrier()
            if rank == 0:
                split = np.array_split(squeezed_cch_array, Nnodes)
            else:
                split = None
            # cch_per_node = int(np.ceil(float(len(squeezed_cch_array)) / Nnodes))
            split_cchs = comm.scatter(split, root=0)
        else:
            split_cchs = squeezed_cch_array
            # pair_per_node = len(squeezed_cch_array)

        # color_array_inst = np.zeros((cch_per_node, B), dtype=int)
        color_array_inst = np.array([], dtype=int)
        pair_tau_ids = np.array([0, 0, 0], dtype=int)
        for count, cchplot in enumerate(split_cchs):
            mask = (cchplot >= threshold)
            int_color_cch = (cchplot[mask] - threshold) / \
                            (1. - threshold) * 10.
            i, j = pair_ids[count]
            if binnums[mask].size:
                pair_tau_ids = np.vstack((pair_tau_ids,
                                          np.array([(i, j, t)
                                                    for t in binnums[mask]])))
                color_array_inst = np.append(color_array_inst,
                                             int_color_cch.astype(int))
        if mpi:
            color_array = comm.gather(color_array_inst, root=0)
            if rank == 0:
                pop_color_array = color_array
        else:
            pop_color_array = color_array_inst

        return pop_color_array, pair_tau_ids[1:]

    def plot_cch_space(self, model, threshold=.05,
                       palette=sns.cubehelix_palette(10, start=.3, rot=.6),
                       alpha=False, **kwargs):
        # color_array is an sparse int array of the thresholded cchs
        # transformed to [0..9] -> 0 is transparent, 1-9 used for indexing the
        # color palette. Since the color_cch is rescaled there is exactly one
        # element with value 10, which always projected to k
        cch_array = self.generate_cch_array(model.spiketrains, maxlag=None,
                                            model=model, **kwargs)
        color_array, pair_tau_ids = self._calc_color_array(cch_array,
                                                           threshold=threshold,
                                                           **kwargs)
        colorarray = np.squeeze(color_array)
        N = len(model.spiketrains)
        B = len(model.spiketrains[0])
        max_cc = np.max(cch_array)
        palette = palette + [[0, 0, 0]]  # single max value is black
        bin_size = self.params['bin_size']

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('Neuron i')
        ax.set_xlim3d(0, N)
        ax.set_ylabel(r'$\tau$ [ms]')
        ax.set_ylim3d(-B / 2 * float(bin_size), B / 2 * float(bin_size))
        ax.set_zlabel('Neuron j')
        ax.set_zlim3d(0, N)

        tau = (np.arange(B) - B / 2) * float(bin_size)
        for count, (i, j, t) in enumerate(pair_tau_ids):
            if alpha:
                color = _alpha(palette[colorarray[count]],
                               1. - colorarray[count] / 10.)
            else:
                color = palette[colorarray[count]]
            if t < 1:
                t = 1
                print('border value shifted')
            elif t == B - 1:
                t = B - 2
                print('border value shifted')
            try:
                ax.plot([j, j], tau[t - 1:t + 1], [i, i], c=color)
            except:
                print('value dropped')
            # expects the outer most values for tau not to be significant

        cax = plt.gcf().add_subplot(222, aspect=10, anchor=(1.1, .5))
        # cmap = colors.ListedColormap(palette)
        # ticks = np.around(np.linspace(threshold, max_cc, 11))
        # cb = colorbar.ColorbarBase(cax, cmap=cmap, orientation='vertical')
        cax.yaxis.set_visible(True)
        cax.yaxis.set_ticks([0, 1])
        print(max_cc)
        cax.set_yticklabels(['{:.2f}'.format(threshold),
                             '{:.2f}'.format(max_cc)])
        return ax, palette

    def draw_pop_cch(self, model, hist_filter=None, color=None,
                     bins=100, figsize=8, **kwargs):
        if color is None:
            color = sns.color_palette()[0]
        cch_array = self.generate_cch_array(model.spiketrains, maxlag=None,
                                            model=model, **kwargs)
        ccharray = copy(np.squeeze(cch_array))
        N = len(ccharray)
        B = len(ccharray[0])
        bin_size = self.params['bin_size']
        w = B / 2 * float(bin_size)
        tau = np.array(list(np.linspace(-w, w, B / 2 * 2 + 1)) * N)
        if hist_filter is None:
            popcch = np.sum(ccharray, axis=0)
            ccharray = ccharray.flatten()
        else:
            if hist_filter == 'max':
                max_array = np.amax(ccharray, axis=1)
                for i, cchval in enumerate(ccharray):
                    ccharray[i] = np.where(cchval < max_array[i], 0,
                                           max_array[i])
            if hist_filter[:9] == 'threshold':
                th = float(hist_filter[9:])
                for i, cchval in enumerate(ccharray):
                    ccharray[i] = np.where(cchval < th, 0, cch)
            popcch = np.sum(ccharray, axis=0)
            ccharray = ccharray.flatten()
            tau = tau[np.where(ccharray)[0]]
            ccharray = ccharray[np.where(ccharray)[0]]
        # grid = sns.jointplot(tau, ccharray, kind=kind, xlim=(-w, w),
        #                      edgecolor="white")
        grid = sns.JointGrid(x=tau, y=ccharray, size=figsize)
        grid.plot_joint(plt.scatter, edgecolor="white", color=color)
        if hist_filter is not None and hist_filter[:9] == 'threshold':
            ax = plt.gca()
            ax.text(-1.2, .9, 'Threshold = {}'.format(th),
                    transform=ax.transAxes)
        grid.set_axis_labels(xlabel=r'$\tau$ [ms]',
                             ylabel='Cross-Correlation Coefficient')
        ax = plt.gca()
        ax.set_xlim((-w, w))
        xvalues = np.linspace(-w, w, B)
        grid.ax_marg_x.bar(xvalues, popcch,
                           width=(xvalues[1]-xvalues[0])*.9,
                           color=color, edgecolor='w')
        hist, edges = np.histogram(ccharray, bins=bins)
        dy = (edges[1]-edges[0])/2.
        yvalues = edges[:-1] + dy
        grid.ax_marg_y.barh(yvalues, hist, height=dy*1.8,
                            color=color, edgecolor='w')
        return grid

    def compute_score(self, observation, prediction):
        score = self.score_type.compute(observation, prediction, **self.params)
        return score
