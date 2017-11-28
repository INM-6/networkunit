from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_correlation_matrix import plot_correlation_matrix
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from quantities import ms
import numpy as np


class generalized_correlation_matrix_test(correlation_test):
    """
    Test to compare the pairwise correlations of a set of neurons in a network.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    params = {'maxlag': 100, # in bins
              'binsize': 2*ms,
              'time_reduction': 'threshold 0.13'
                }

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        # check is model has already stored prediction
        spiketrains = model.produce_spiketrains(**self.params)
        cch_array  = self.generate_cch_array(spiketrains=spiketrains,
                                       **self.params)
        pairs_idx = np.triu_indices(len(spiketrains), 1)
        pairs = [[i, j] for i, j in zip(pairs_idx[0], pairs_idx[1])]
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
            th_cch_array = np.array([a[a>th] for a in np.squeeze(cch_array)])
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
        cc_mat = np.ones((dim,dim))
        for count, (i,j) in enumerate(pair_ids):
            cc_mat[i,j] = cc_array[count]
            cc_mat[j,i] = cc_array[count]
        return cc_mat

    def visualize_sample(self, model1=None, model2=None, ax=None,
                         palette=None,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', **kwargs):

        matrices, palette = self._create_plotting_samples(model1=model1,
                                                          model2=model2,
                                                          palette=palette)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plot_correlation_matrix(matrices[0], ax=ax[0], remove_autocorr=True,
                                labels=None, sort=False, cluster=False,
                                linkmethod='ward', dendrogram_args={},
                                **kwargs)
        ax[0].set_title(sample_names[0])
        plot_correlation_matrix(matrices[1], ax=ax[1], remove_autocorr=True,
                                labels=None, sort=False, cluster=False,
                                linkmethod='ward', dendrogram_args={},
                                **kwargs)
        ax[1].set_title(sample_names[1])
        return ax


