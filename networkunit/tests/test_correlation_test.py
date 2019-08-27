from elephant.spike_train_correlation import corrcoef, cch
from elephant.conversion import BinnedSpikeTrain
import neo
import numpy as np
from quantities import ms, quantity
from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots import alpha as _alpha

class correlation_test(two_sample_test):
    """
    Abstract test class  to compare the pairwise correlations between spike
    trains of a set of neurons in a network.

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

    def __init__(self, observation=None, name=None, **params):
        if params is None:
            params = {}
        self.params.update(params)
        if 'binsize' not in self.params and 'num_bins' not in self.params:
            self.params['binsize'] = 2*ms
        super(correlation_test, self).__init__(observation=observation,
                                               name=name, **self.params)

    def validate_observation(self, observation):
        # ToDo: Check if observation values are legit (non nan, positive, ...)
        pass

    def robust_BinnedSpikeTrain(self, spiketrains, binsize=None, num_bins=None,
                                t_start=None, t_stop=None, **add_args):
        if type(spiketrains) == neo.core.spiketrain.SpikeTrain:
            spiketrains = [spiketrains]
        if t_start is None:
            t_start = min([st.t_start for st in spiketrains])
        if t_stop is None:
            t_stop = min([st.t_stop for st in spiketrains])
        if binsize is None and num_bins is None:
            binsize = self.params['binsize']
        return BinnedSpikeTrain(spiketrains, binsize=binsize,
                                num_bins=num_bins, t_start=t_start,
                                t_stop=t_stop)

    def generate_correlations(self, spiketrains=None, binary=False,
                              nan_to_num=False, **kwargs):
        cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                            binary=binary, **kwargs)
        if nan_to_num:
            cc_matrix = np.nan_to_num(cc_matrix)
        idx = np.triu_indices(len(cc_matrix), 1)
        return cc_matrix[idx]

    def generate_cc_matrix(self, spiketrains=None, binary=False, model=None,
                           nan_to_num=False, **kwargs):
        """
        Calculates the covariances between all pairs of spike trains.

        Parameters
        ----------
        spiketrain_list : list of neo.SpikeTrain (default None)
            If no list is passed the function tries to access the class
            parameter 'spiketrains'.

        binary: bool (default False)
            Parameter is passed to
            elephant.spike_train_correlation.covariance()

        kwargs:
            Passed to elephant.conversion.BinnedSpikeTrain()

        Returns : list of floats
            list of covariances of length = (N^2 - N)/2 where N is the number
            of spike trains.
        -------
        """
        binned_sts = self.robust_BinnedSpikeTrain(spiketrains, **kwargs)

        cc_matrix = corrcoef(binned_sts, binary=binary)

        if nan_to_num:
            cc_matrix = np.nan_to_num(cc_matrix)
        return cc_matrix
