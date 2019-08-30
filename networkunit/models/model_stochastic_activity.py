import sciunit
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
import numpy as np
from elephant.spike_train_generation import single_interaction_process as SIP
from elephant.spike_train_generation import compound_poisson_process as CPP
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import ms, Hz, quantity
from networkunit.plots.plot_rasterplot import rasterplot
import neo
import random


class stochastic_activity(sciunit.Model, ProducesSpikeTrains):
    """
    Model class which is able to generate stochastic spiking data

    Parameters
    ----------
        size : int
            Number of spike trains
        t_start : quantity
            starting time
        t_stop : quantity
            ending time
        rate : quantity
            average firing rate
        statistic : 'poisson', 'gamma'(to be implemented)
        correlation_method: 'CPP', 'spatio-temporal', 'pairwise_equivalent', None
            CPP - compound Poisson process generating correlated activity of
                  size 'assembly_sizes' with mean correlation according to
                  'correlations'.
            spatio-temporal - generates CPP correlated groups and shifts the
                              spike trains randomly within +- 0.5*'max_pattern_length'
                              to create spatio-temporal patterns.
            pairwise_equivalent - generates pairs of correlated spike trains
                                  so that the amount of correlation is equivalent
                                  to a correlated group with parameters
                                  'assembly_sizes' and 'correlations'
        expected_binsize : quantity
            Binsize with which correlations are calculated to be able to
            generate the pairwise equivalent.
        correlations : float, list of floats
            Average correlation for the correlated group(s). Pass a list of
            floats if there are multiple groups with different correlations.
            If 0, it generates  homogenous Poisson activity.
        assembly_sizes : list of ints
            Size(s) of correlated group(s). Empty list for no correlations.
        bkgr_correlation : float
            Background correlation (to be implemented)
        max_pattern_length : quantity
            Maximum pattern length for spatio-temporal patterns.
        shuffle : bool
            Shuffle the spike trains to separate the correlated groups
        shuffle_seed : int

    Example
    ----------

    """
    params = {'size': 100,
              't_start': 0 * ms,
              't_stop': 10000 * ms,
              'rate': 10 * Hz,
              'statistic': 'poisson',
              'correlation_method': 'CPP', # 'spatio-temporal', 'pairwise_equivalent'
              'expected_binsize': 2 * ms,
              'correlations': 0.,
              'assembly_sizes': [],
              'bkgr_correlation': 0.,
              'max_pattern_length':100 * ms,
              'shuffle': False,
              'shuffle_seed': None}

    def __init__(self, name=None, backend='storage', **params):
        self.params.update(params)
        # updating params is only for testing reasons
        # for usage in the validation framework, the params need to be fixed!
        self.__dict__.update(self.params)
        self.check_input()
        super(stochastic_activity, self).__init__(name=name, **self.params)
        if backend is not None:
            self.set_backend(backend)

    def check_input(self):
        if not type(self.correlations) == list:
            self.correlations = [self.correlations] * len(self.assembly_sizes)
        elif len(self.correlations) == 1:
            self.correlations *= len(self.assembly_sizes)
        if len(self.assembly_sizes) == 1 and self.assembly_sizes[0] == 1:
            self.assembly_sizes = []
        pass

    def produce_spiketrains(self, **kwargs):
        if not self.spiketrains:
            self.spiketrains = self.generate_spiketrains(**kwargs)
        return self.spiketrains

    def get_backend(self):
        """Return the simulation backend."""
        return self._backend

    def set_backend(self, backend):
        """Set the simulation backend."""
        if isinstance(backend, str) and backend in available_backends:
            self.backend = backend
            self._backend = available_backends[backend]()
        elif backend is None:
            # The base class should not be called.
            raise Exception(("A backend must be selected"))
        else:
            raise Exception("Backend %s not found in backends" % name)
        self._backend.model = self
        self._backend.init_backend(*args, **kwargs)

    def generate_spiketrains(self, **kwargs):
        spiketrains = [None] * self.size

        if self.correlation_method == 'pairwise_equivalent':
        # change input to pairwise correlations with expected distribution
        # correlation coefficients
            nbr_of_pairs = [0] * len(self.assembly_sizes)
            new_correlation = []
            for i, A_size in enumerate(self.assembly_sizes):
                nbr_of_pairs[i] = int(A_size * (A_size - 1) / 2.)
                new_correlation = new_correlation + [self.correlations[i]]*nbr_of_pairs[i]
            if sum(nbr_of_pairs)*2 > self.size:
                raise ValueError('Assemblies are too large to generate an ' +
                                 'pairwise equivalent with the network size.')
            self.assembly_sizes = [2] * sum(nbr_of_pairs)
            self.correlations = new_correlation
            self.correlation_method = 'CPP'

        # generate correlated assemblies
        for i, a_size in enumerate(self.assembly_sizes):
            if a_size < 2:
                raise ValueError('An assembly must consists of at least two units.')
            generated_sts = int(np.sum(self.assembly_sizes[:i]))
            spiketrains[generated_sts:generated_sts + a_size] \
                = self._generate_assembly(correlation=self.correlations[i],
                                          A_size=a_size)
            for j in range(a_size):
                spiketrains[generated_sts + j].annotations = {'Assembly': str(i)}

        # generate background
        if self.bkgr_correlation > 0:
            dummy = None
            # ToDo: background generation without cpp
        else:
            spiketrains[sum(self.assembly_sizes):] \
                = np.array([HPP(rate=self.rate, t_start=self.t_start, t_stop=self.t_stop)
                            for _ in range(self.size - sum(self.assembly_sizes))])

        if self.shuffle:
            if self.shuffle_seed is None:
                random.shuffle(spiketrains)
            else:
                random.Random(self.shuffle_seed).shuffle(spiketrains)

        for i in range(self.size):
            spiketrains[i].annotations['Model'] = self.name
        return spiketrains

    def _generate_assembly(self, correlation, A_size, **kwargs):

        syncprob = self._correlation_to_syncprob(cc=correlation,
                                                 A_size=A_size,
                                                 rate=self.rate,
                                                 T=self.t_stop - self.t_start,
                                                 binsize=self.expected_binsize)
        bkgr_syncprob = self._correlation_to_syncprob(cc=self.bkgr_correlation,
                                                      A_size=2,
                                                      rate=self.rate,
                                                      T=self.t_stop - self.t_start,
                                                      binsize=self.expected_binsize)
        if self.correlation_method == 'CPP' \
        or self.correlation_method == 'spatio-temporal':
            assembly_sts = self._generate_CPP_assembly(A_size=A_size,
                                                       syncprob=syncprob,
                                                       bkgr_syncprob=bkgr_syncprob)
            if self.correlation_method == 'CPP':
                return assembly_sts
            else:
                return self._shift_spiketrains(assembly_sts)
        else:
            raise NameError("Method name not known!")

    def _generate_CPP_assembly(self, A_size, syncprob, bkgr_syncprob):
        amp_dist = np.zeros(A_size + 1)
        amp_dist[1] = 1. - syncprob - bkgr_syncprob
        amp_dist[2] = bkgr_syncprob
        amp_dist[A_size] = syncprob
        np.testing.assert_almost_equal(sum(amp_dist), 1., decimal=4)
        amp_dist *= (1. / sum(amp_dist))
        ref_rate = self.rate #* A_size / float(1+syncprob*(A_size-1))
        # The CPP already takes the expected rate and not the reference rate
        return CPP(rate=self.rate, A=amp_dist,
                   t_start=self.t_start, t_stop=self.t_stop)

    def _shift_spiketrains(self, assembly_sts):#
        shifted_assembly_sts = [None] * len(assembly_sts)
        for i, st in enumerate(assembly_sts):
            spiketimes = np.array(st.tolist())
            shift = np.random.rand() * self.max_pattern_length \
                                     - self.max_pattern_length
            shift = float(shift.rescale('ms'))
            spiketimes = spiketimes + shift
            pos_fugitives = np.where(spiketimes >= float(self.t_stop))[0]
            neg_fugitives = np.where(spiketimes <= float(self.t_start))[0]
            spiketimes[pos_fugitives] = spiketimes[pos_fugitives] - float(
                self.t_stop)
            spiketimes[neg_fugitives] = spiketimes[neg_fugitives] + float(
                self.t_stop - self.t_start)
            shifted_assembly_sts[i] = neo.SpikeTrain(times=sorted(spiketimes),
                                                     units='ms',
                                                     t_start=self.t_start,
                                                     t_stop=self.t_stop)
        return shifted_assembly_sts

    def _correlation_to_syncprob(self, cc, A_size, rate, T, binsize):
        if A_size < 2:
            raise ValueError
        if cc == 1.:
            return 1.
        if not cc:
            return 0.
        # m0 = rate * T / (float(T)/float(binsize))
        m = rate * binsize
        if type(m) == quantity.Quantity:
            m = float(m.rescale('dimensionless'))
        A = float(A_size)
        # root = np.sqrt((cc*n -cc-n)**2 + 4*m0*(cc*n-cc-n+1))
        # adding = (n-1)*(-2*cc*m0 + cc*n + 2*m0) - n**2
        # denominator = 2 * (cc - 1.) * m0 * (n - 1.) ** 2
        # sync_prob = (n * root + adding) / denominator
        sync_prob = (1 - (m-1)*(cc-1))/(1+ (m-1)*(cc-1)*(A-1))
        if type(sync_prob) == quantity.Quantity:
            if bool(sync_prob.dimensionality):
                raise ValueError
            else:
                return float(sync_prob.magnitude)
        else:
            return sync_prob

    def show_rasterplot(self, **kwargs):
        return rasterplot(self.spiketrains, **kwargs)


# Todo: Handle quantity inputs which are not ms or Hz
