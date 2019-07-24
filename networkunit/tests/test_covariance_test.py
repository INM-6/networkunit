from elephant.spike_train_correlation import covariance
from elephant.conversion import BinnedSpikeTrain
from numpy import triu_indices
from quantities import ms
from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains


class covariance_test(two_sample_test):
    """
    Test to compare the pairwise covariances of a set of neurons in a network.
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
    binary: bool
        If true, the binned spike trains are set to be binary.
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        covariances = self.get_prediction(model)
        if covariances is None:
            if kwargs:
                self.params.update(kwargs)
            if 'binsize' not in self.params and 'num_bins' not in self.params:
                self.params['binsize'] = 2*ms
            self.spiketrains = model.produce_spiketrains(**self.params)
            covariances = self.generate_covariances(self.spiketrains,
                                                    **self.params)
            self.set_prediction(model, covariances)
        return covariances

    def validate_observation(self, observation):
        # ToDo: Check if observation values are legit (non nan, positive, ...)
        pass

    def generate_covariances(self, spiketrain_list=None, binary=False,
                             ** kwargs):
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
        def robust_BinnedSpikeTrain(spiketrains, binsize=None, num_bins=None,
                                    t_start=None, t_stop=None, **add_args):
            return BinnedSpikeTrain(spiketrains, binsize=binsize,
                                    num_bins=num_bins, t_start=t_start,
                                    t_stop=t_stop)
        if spiketrain_list is None:
            # assuming the class has the property 'spiketrains' and it
            # contains a list of neo.Spiketrains
            binned_sts = robust_BinnedSpikeTrain(self.spiketrains, **kwargs)
        else:
            binned_sts = robust_BinnedSpikeTrain(spiketrain_list, **kwargs)
        cov_matrix = covariance(binned_sts, binary=binary)
        idx = triu_indices(len(cov_matrix), 1)
        return cov_matrix[idx]
