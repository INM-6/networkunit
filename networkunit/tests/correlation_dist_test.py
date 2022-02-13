from networkunit.tests.correlation_test import correlation_test
from networkunit.tests.correlation_test import correlation_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import use_cache


class correlation_dist_test(correlation_test):
    """
    Abstract test class to compare the distributions of pairwise correlations
    of a set of spiking neurons in a network.
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
        Start of time window used to calculate the correlation coefficents.
    t_stop: quantity, None
        Stop of time window used to calculate the correlation coefficents.
    nan_to_num: bool
        If true, np.nan are set to 0, and np.inf to largest finite float.
    binary: bool
        If true, the binned spike trains are set to be binary.
    """

    required_capabilities = (ProducesSpikeTrains, )

    default_params = {**correlation_test.default_params,
                      }

    @use_cache
    def generate_prediction(self, model):
        spiketrains = model.produce_spiketrains(**self.params)
        cc_samples = self.generate_correlations(spiketrains=spiketrains)
        return cc_samples
