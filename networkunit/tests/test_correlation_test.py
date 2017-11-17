from elephant.spike_train_correlation import corrcoef, cch
from elephant.conversion import BinnedSpikeTrain
import numpy as np
from quantities import ms, quantity
from networkunit.tests import two_sample_test
from networkunit.capabilities import ProducesSpikeTrains
from abc import ABCMeta, abstractmethod


class correlation_test(two_sample_test):
    """
    Test to compare the pairwise correlations of a set of neurons in a network.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    params = {'maxlag': 100, # in bins
                }

    def __init__(self, observation=None, name=None, **params):
        if params is None:
            params = {}
        self.params.update(params)
        if 'binsize' not in self.params and 'num_bins' not in self.params:
            self.params['binsize'] = 2*ms
        super(correlation_test, self).__init__(observation=observation,
                                               name=name, **params)

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        # check is model has already stored prediction
        spiketrains = model.produce_spiketrains(**self.params)
        return self.generate_correlations(spiketrains=spiketrains,
                                         **self.params)

    def validate_observation(self, observation):
        # ToDo: Check if observation values are legit (non nan, positive, ...)
        pass

    def robust_BinnedSpikeTrain(self, spiketrains, binsize=None, num_bins=None,
                                t_start=None, t_stop=None, **add_args):
        if t_start is None:
            t_start = min([st.t_start for st in spiketrains])
        if t_stop is None:
            t_stop = min([st.t_stop for st in spiketrains])
        if binsize is None and num_bins is None:
            binsize = self.params['binsize']
        return BinnedSpikeTrain(spiketrains, binsize=binsize,
                                num_bins=num_bins, t_start=t_start,
                                t_stop=t_stop)

    def generate_correlations(self, spiketrains=None, binary=False, **kwargs):
        self.generate_cc_matrix(spiketrains=spiketrains,
                                binary=binary, **kwargs)
        idx = np.triu_indices(len(self.cc_matrix), 1)
        return self.cc_matrix[idx]

    def generate_cc_matrix(self, spiketrains=None, binary=False, **kwargs):
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
        # if spiketrains is None:
        #     binned_sts = self.robust_BinnedSpikeTrain(self.spiketrains, **kwargs)
        # else:
        binned_sts = self.robust_BinnedSpikeTrain(spiketrains, **kwargs)

        cc_matrix = corrcoef(binned_sts, binary=binary)
        return cc_matrix

    def generate_cch_array(self, spiketrains, maxlag=None,
                           **kwargs):
        if not hasattr(self, 'cch_array'):
            return self.cch_array
        else:
            if 'binsize' in self.params:
                binsize = self.params['binsize']
            else:
                t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
                tmin = min(t_lims, key=lambda f: f[0])[0]
                tmax = max(t_lims, key=lambda f: f[1])[1]
                T = tmax - tmin
                binsize = T / float(self.params['num_bins'])
            if maxlag is None:
                maxlag = self.params['maxlag']
            else:
                self.params['maxlag'] = maxlag
            if type(maxlag) == quantity.Quantity:
                maxlag = int(float(maxlag.rescale('ms'))
                           / float(binsize.rescale('ms')))
            try:
                from mpi4py import MPI
                mpi = True
            except:
                mpi = False
            N = len(spiketrains)
            B = 2 * maxlag + 1
            pairs_idx = np.triu_indices(N, 1)
            pairs = [[i, j] for i, j in zip(pairs_idx[0], pairs_idx[1])]
            if mpi:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                Nnodes = comm.Get_size()
                comm.Barrier()
                if rank == 0:
                    split = np.array_split(pairs, Nnodes)
                else:
                    split = None
                pair_per_node = int(np.ceil(float(len(pairs)) / Nnodes))
                split_pairs = comm.scatter(split, root=0)
            else:
                split_pairs = pairs
                pair_per_node = len(pairs)

            cch_array = np.zeros((pair_per_node, B))
            max_cc = 0
            for count, (i, j) in enumerate(split_pairs):
                binned_sts_i = self.robust_BinnedSpikeTrain(spiketrains[i],
                                                            binsize=binsize)
                binned_sts_j = self.robust_BinnedSpikeTrain(spiketrains[j],
                                                            binsize=binsize)
                cch_array[count] = np.squeeze(cch(binned_sts_i,
                                                  binned_sts_j,
                                                  window=[-maxlag, maxlag],
                                                  cross_corr_coef=True,
                                                  **kwargs)[0])
                max_cc = max([max_cc, max(cch_array[count])])
            if mpi:
                pop_cch = comm.gather(cch_array, root=0)
                pop_max_cc = comm.gather(max_cc, root=0)
                if rank == 0:
                    cch_array = pop_cch
                    max_cc = pop_max_cc
            self.cch_array = cch_array
            self.max_cc = max_cc
            return self.cch_array
