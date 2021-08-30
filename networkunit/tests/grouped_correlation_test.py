from networkunit.tests.correlation_test import correlation_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
import numpy as np


class grouped_correlation_test(correlation_test):
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
    type: list of str
        Type of metrics that should be measured from the correlation cc_matrix
        Default: ['avg', 'std']
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        preds = self.get_prediction(model)
        if preds is None:
            if kwargs:
                self.params.update(kwargs)
            lists_of_spiketrains = model.produce_grouped_spiketrains(**self.params)

            if 'metrics' in self.params.keys():
                metrics = self.params.pop('metrics')
                if not isinstance(metrics, list):
                    metrics = [metrics]
            else:
                metrics = ['avg', 'std']

            avg_correlations = np.array([])
            std_correlations = np.array([])

            for sts in lists_of_spiketrains:
                if len(sts) == 1:
                    correlation_avgs = np.array([np.nan])
                    correlation_stds = np.array([np.nan])
                else:
                    cc_matrix = self.generate_cc_matrix(spiketrains=sts,
                                                        model=model,
                                                        **self.params)
                    np.fill_diagonal(cc_matrix, np.nan)

                    if 'avg' in metrics:
                        correlation_avgs = \
                            np.nansum(cc_matrix, axis=0) / len(sts)
                    if 'std' in metrics:
                        correlation_stds = np.nanstd(cc_matrix, axis=0)


                if 'avg' in metrics:
                    avg_correlations = np.append(avg_correlations,
                                                 correlation_avgs)
                if 'std' in metrics:
                    std_correlations = np.append(std_correlations,
                                                 correlation_stds)


            if self.params['nan_to_num']:
                if 'avg' in metrics:
                    avg_correlations = np.nan_to_num(avg_correlations)
                if 'std' in metrics:
                    std_correlations = np.nan_to_num(std_correlations)

            if ('avg' in metrics) and ('std' not in metrics):
                preds = avg_correlations
            if ('avg' not in metrics) and ('std' in metrics):
                preds = std_correlations
            if ('avg' in metrics) and ('std' in metrics):
                preds = np.array([avg_correlations, std_correlations]).T

            self.set_prediction(model, preds)

        return preds
