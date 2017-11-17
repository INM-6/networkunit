from elephant.spike_train_correlation import covariance
from elephant.conversion import BinnedSpikeTrain
from numpy import triu_indices
from quantities import ms
import sciunit

########## Deprecated! ##########

class ProducesCovariances(sciunit.Capability):
    """
    A capability to produce a sample of pairwise covariances between
    spiketrains of a network.
    """
    def produce_covariances(self, spiketrain_list=None, binary=False,
                            **kwargs):
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
        try:
            def robust_BinnedSpikeTrain(spiketrains, binsize=2*ms, num_bins=None,
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
        except:
            self.unimplemented()