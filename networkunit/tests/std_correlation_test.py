from networkunit.tests.correlation_test import correlation_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import use_cache, parallelize
import numpy as np

class std_correlation_test(correlation_test):
    """
    Abstract test class to compare correlation matrices of a set of
    spiking neurons in a network.
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
        Start of time window used to calculate the correlation coefficients.
    t_stop: quantity, None
        Stop of time window used to calculate the correlation coefficients.
    nan_to_num: bool
        If true, np.nan are set to 0, and np.inf to largest finite float.
    binary: bool
        If true, the binned spike trains are set to be binary.
    """

    required_capabilities = (ProducesSpikeTrains, )

    @use_cache
    def generate_prediction(self, model):
        lists_of_spiketrains = model.produce_grouped_spiketrains(**self.params)

        with parallelize(self.calc_std_correlations, self) as std_corr_parallel:
            std_correlations = std_corr_parallel(lists_of_spiketrains)

        std_correlations = np.concatenate(std_correlations)

        if self.params['nan_to_num']:
            std_correlations = np.nan_to_num(std_correlations)

        return std_correlations


    def calc_std_correlations(self, spiketrains):
        if len(spiketrains) == 1:
            std_corr = np.array([np.nan])
        else:
            cc_matrix = self.generate_cc_matrix(spiketrains)
            np.fill_diagonal(cc_matrix, np.nan)

            std_corr = np.nanstd(cc_matrix, axis=0)
        return std_corr
