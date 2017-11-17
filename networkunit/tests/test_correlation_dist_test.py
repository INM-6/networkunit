from networkunit.tests import correlation_test
from networkunit.capabilities import ProducesSpikeTrains
from abc import ABCMeta, abstractmethod


class correlation_dist_test(correlation_test):
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
        return self.generate_correlations(spiketrains=spiketrains,
                                          **self.params)


