from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains


class correlation_dist_test(correlation_test):
    """
    Abstract test class to compare the distributions of pairwise correlations
    of a set of spiking neurons in a network.
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
        Start of time window used to calculate the correlation coefficents.
    t_stop: quantity, None
        Stop of time window used to calculate the correlation coefficents.
    nan_to_num: bool
        If true, np.nan are set to 0, and np.inf to largest finite float.
    binary: bool
        If true, the binned spike trains are set to be binary.
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        cc_samples = self.get_prediction(model)
        if cc_samples is None:
            if kwargs:
                self.params.update(kwargs)
            spiketrains = model.produce_spiketrains(**self.params)
            cc_samples = self.generate_correlations(spiketrains=spiketrains,
                                                    **self.params)
            self.set_prediction(model, cc_samples)
        return cc_samples
