from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import mean_firing_rate


class firing_rate_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        rates = self.get_prediction(model)
        if rates is None:
            if kwargs:
                self.params.update(kwargs)
            spiketrains = model.produce_spiketrains(**self.params)
            rates = [mean_firing_rate(st).rescale('Hz') for st in spiketrains]
            self.set_prediction(model, rates)
        return rates
