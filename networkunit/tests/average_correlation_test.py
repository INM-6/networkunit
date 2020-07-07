from networkunit.tests.correlation_test import correlation_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
import numpy as np


class average_correlation_test(correlation_test):
    """
    Abstract test class to compare correlation matrices of a set of
    spiking neurons in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.

    Parameters (in dict params):
    ----------
    binsize: quantity, None (default: 2*ms)
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

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        avg_correlations = self.get_prediction(model)
        if avg_correlations is None:
            if kwargs:
                self.params.update(kwargs)
            lists_of_spiketrains = model.produce_grouped_spiketrains(**self.params)
            avg_correlations = np.array([])

            for sts in lists_of_spiketrains:
                if len(sts) == 1:
                    correlation_averages = np.array([np.nan])
                else:
                    cc_matrix = self.generate_cc_matrix(spiketrains=sts,
                                                        model=model,
                                                        **self.params)
                    np.fill_diagonal(cc_matrix, 0.)

                    correlation_averages = np.nansum(cc_matrix,
                                                     axis=0) / len(sts)
                avg_correlations = np.append(avg_correlations,
                                             correlation_averages)

            if self.params['nan_to_num']:
                avg_correlations = np.nan_to_num(avg_correlations)
            self.set_prediction(model, avg_correlations)
        return avg_correlations
