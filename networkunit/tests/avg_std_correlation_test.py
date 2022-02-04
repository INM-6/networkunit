from networkunit.tests.correlation_test import correlation_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import use_cache, parallelize
import numpy as np

class avg_std_correlation_test(correlation_test):
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

    default_params = {**correlation_test.default_params,
                      'statistic': 'avg'}

    @use_cache
    def generate_prediction(self, model):

        spiketrains_lst = model.produce_grouped_spiketrains(**self.params)

        cc_matrix_lst = self.generate_cc_matrix(model, spiketrains=spiketrains_lst)

        predictions = []
        for cc_matrix in cc_matrix_lst:
            np.fill_diagonal(cc_matrix, np.nan)
            avg_corr = np.nansum(cc_matrix, axis=0) / (cc_matrix.shape[0] - 1)
            if self.params['statistic'] == 'std':
                std_corr = np.sqrt(np.nansum(
                    (cc_matrix-avg_corr[np.newaxis, :])**2, axis=0) /
                    (cc_matrix.shape[0] - 1))
                predictions.append(std_corr)
            elif self.params['statistic'] == 'avg':
                predictions.append(avg_corr)

        predictions = np.concatenate(predictions)

        if self.params['nan_to_num']:
            predictions = np.nan_to_num(predictions)

        return predictions
