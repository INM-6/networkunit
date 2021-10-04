from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from quantities import ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import spike_train_timescale as timescale
import numpy as np
from networkunit.utils import generate_prediction_wrapper

class timescale_test(two_sample_test):
    """
    Test to compare the timescales a set of spiking neurons in a network.
    The timescale is defined as the decay of the autocorrelation function of a
    given spike train. The timescale is returned in 'ms'.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.

    Parameters (in dict params):
    ----------
    bin_size: quantity (default: 1*ms)
        Size of bins used to calculate the spiketrain timescale.
    tau_max: quantity (default: 100*ms)
        Maximal integration time of the auto-correlation function.
    min_spikecount: int (default: 2)
        Minimum number of spikes required to compute the timescale, if less
        spikes are found np.nan is returned.

    """

    name = 'Timescale'
    required_capabilities = (ProducesSpikeTrains, )
    default_params = {'bin_size': 1*ms,
                      'tau_max': 100*ms,
                      'min_spikecount': 2}

    @generate_prediction_wrapper
    def generate_prediction(self, model, **params):
        spiketrains = model.produce_spiketrains(**params)

        tau_list = []
        for st in spiketrains:
            if len(st.times) >= params['min_spikecount']:
                bst = BinnedSpikeTrain(st, bin_size=params['bin_size'])
                tau = timescale(bst, params['tau_max'])
                tau_list.append(tau.rescale('ms'))
            else:
                tau_list.append(np.nan)

        return tau_list
