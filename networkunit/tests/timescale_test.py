from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from quantities import ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import spike_train_timescale as timescale


class timescale_test(two_sample_test):
    """
    Test to compare the timescales a set of spiking neurons in a network.
    The timescale is defined as the decay of the autocorrelation function of
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.

    Parameters (in dict params):
    ----------
    binsize: quantity, None (default: 1*ms)
        Size of bins used to calculate the spiketrain timescale.
    tau_max: quantity, None (default: 100*ms)
        Maximal integration time of the auto-correlation function.
    """

    name = 'Timescale'
    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        tau_list = self.get_prediction(model)
        if tau_list is None:
            if kwargs:
                self.params.update(kwargs)
            if 'binsize' not in self.params:
                self.params['binsize'] = 1*ms
            if 'tau_max' not in self.params:
                self.params['tau_max'] = 100*ms
            spiketrains = model.produce_spiketrains(**self.params)
            binned_sts = [BinnedSpikeTrain(st, binsize=self.params['binsize'])
                          for st in spiketrains]
            tau_list = [timescale(st, self.params['tau_max']).rescale('ms')
                        for st in binned_sts]
            self.set_prediction(model, tau_list)
        return tau_list
