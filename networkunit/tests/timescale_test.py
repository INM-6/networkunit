from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from quantities import ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import spike_train_timescale as timescale
import numpy as np
from networkunit.utils import use_cache, parallelize

class timescale_test(two_sample_test):
    """
    Test to compare the timescales a set of spiking neurons in a network.
    The timescale is defined as the decay of the autocorrelation function of a
    given spike train. The timescale is returned in 'ms'.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.

    Parameters:
    ----------
    bin_size: quantity (default: 1*ms)
        Size of bins used to calculate the spiketrain timescale.
    tau_max: quantity (default: 100*ms)
        Maximal integration time of the auto-correlation function.
    min_spikecount: int (default: 2)
        Minimum number of spikes required to compute the timescale, if less
        spikes are found np.nan is returned.

    """

    required_capabilities = (ProducesSpikeTrains, )
    default_params = {**two_sample_test.default_params,
                      'bin_size': 1*ms,
                      'tau_max': 100*ms,
                      'min_spikecount': 2}

    @use_cache
    def generate_prediction(self, model):
        spiketrains = model.produce_spiketrains(**self.params)

        with parallelize(self.calc_timescales, self) as calc_timescales_parallel:
            timescales = calc_timescales_parallel

        return np.array(timescales)


    def calc_timescales(self, spiketrain):
        if len(spiketrain.times) >= self.params['min_spikecount']:
            bst = BinnedSpikeTrain(spiketrain, bin_size=self.params['bin_size'])
            tau = timescale(bst, self.params['tau_max'])
            tau = tau.rescale('ms').magnitude
        else:
            tau = np.nan
        return tau
