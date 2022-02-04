from elephant.spike_train_correlation import correlation_coefficient
from elephant.conversion import BinnedSpikeTrain
import neo
import numpy as np
from quantities import ms
from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import filter_valid_params, parallelize, use_cache



class correlation_test(two_sample_test):
    """
    Abstract test class  to compare the pairwise correlations between spike
    trains of a set of neurons in a network.

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
    nan_to_num: bool (default: False)
        If true, np.nan are set to 0, and np.inf to largest finite float.
    binary: bool
        If true, the binned spike trains are set to be binary.
    """

    required_capabilities = (ProducesSpikeTrains, )

    default_params = {**two_sample_test.default_params,
                      'correlation_cache_key': None,
                      'bin_size': 2*ms,
                      'nan_to_num': False,
                      'corrcoef_norm': True}


    def generate_correlations(self, spiketrains=None, model=None):
        cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                            model=model)
        idx = np.triu_indices(len(cc_matrix), 1)
        return cc_matrix[idx]


    @use_cache(cache_key_param='correlation_cache_key')
    def generate_cc_matrix(self, model, spiketrains=None):
        if spiketrains is None:
            spiketrains = model.produce_spiketrains()

        if isinstance(spiketrains[0], list):
            with parallelize(self.calculate_cc_matrix, self) as calc:
                cc_matrix = calc(spiketrains)
        else:
            cc_matrix = self.calculate_cc_matrix(spiketrains)

        return cc_matrix

    def calculate_cc_matrix(self, spiketrains):
        with filter_valid_params(BinnedSpikeTrain) as _BinnedSpikeTrain:
            binned_sts = _BinnedSpikeTrain(spiketrains, **self.params)

        # if 'corrcoef_norm' in self.params and 'corrcoef_norm'
        with filter_valid_params(correlation_coefficient) as _corrcoef:
            cc_matrix = _corrcoef(binned_sts, **self.params)

        if self.params['nan_to_num']:
            cc_matrix = np.nan_to_num(cc_matrix)

        return cc_matrix
