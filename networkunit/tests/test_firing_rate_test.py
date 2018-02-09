from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import mean_firing_rate
from abc import ABCMeta, abstractmethod


class firing_rate_test(two_sample_test):
    """
    Test to compare the firing rates of a set of neurons in a network.
    """
    # __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        if kwargs:
            self.params.update(kwargs)
        spiketrains = model.produce_spiketrains(**self.params)
        rates = [mean_firing_rate(st) for st in spiketrains]
        return rates


