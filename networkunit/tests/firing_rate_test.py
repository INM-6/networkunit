from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import mean_firing_rate
from networkunit.utils import use_cache, parallelize


class firing_rate_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.
    """

    required_capabilities = (ProducesSpikeTrains, )

    @use_cache
    def generate_prediction(self, model):
        spiketrains = model.produce_spiketrains(**self.params)

        def mean_firing_rate_Hz(st):
            return mean_firing_rate(st).rescale('Hz')

        with parallelize(mean_firing_rate_Hz, self) as parallel_mean_firing_rate_Hz:
            rates = parallel_mean_firing_rate_Hz(spiketrains)

        return rates
