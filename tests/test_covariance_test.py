from elephant.spike_train_correlation import covariance
from elephant.conversion import BinnedSpikeTrain
from numpy import triu_indices
from quantities import ms
from networkunit.tests.base_tests.ABCtest_two_sample_test import two_sample_test
from networkunit.capabilities import ProducesSpikeTrains
from abc import ABCMeta, abstractmethod


class covariance_test(two_sample_test):
    """
    Test to compare the pairwise covariances of a set of neurons in a network.
    The statistical testing method needs to be passed in form of a
    sciunit.Score as score_type on initialization.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        if 'binsize' not in self.params and 'num_bins' not in self.params:
            self.params['binsize'] = 2*ms

        self.spiketrains = model.produce_spiketrains(**self.params)
        return self.generate_covariances(spiketrain_list=self.spiketrains,
                                         **self.params)

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
