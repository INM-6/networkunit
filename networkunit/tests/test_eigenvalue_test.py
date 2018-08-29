from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from abc import ABCMeta, abstractmethod
from scipy.linalg import eigh


class eigenvalue_test(correlation_test):
    """
    Test to compare the eigenvalues of correlation matrices of a set of
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
        if kwargs:
            self.params.update(kwargs)
        if not hasattr(model, 'prediction'):
            model.prediction = {}
        if self.test_hash in model.prediction:
            ews = model.prediction[self.test_hash]
        else:
            spiketrains = model.produce_spiketrains(**self.params)
            cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                                **self.params)
            ews, _ = eigh(cc_matrix)
            model.prediction[self.test_hash] = ews
        return ews


