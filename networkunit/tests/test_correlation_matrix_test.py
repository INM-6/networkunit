from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_correlation_matrix import plot_correlation_matrix
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod


class correlation_matrix_test(correlation_test):
    """
    Test to compare the pairwise correlations of a set of neurons in a network.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        # check is model has already stored prediction
        spiketrains = model.produce_spiketrains(**self.params)
        return self.generate_cc_matrix(spiketrains=spiketrains,
                                       **self.params)

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