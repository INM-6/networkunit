import numpy as np
from scipy.linalg import eigh
from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains


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
    dimensionality: bool (default: False)
        If true, compute the dimensionality from the eigenvalues.
        dimensionality is calculated in slices of Lslice length
    Lslice: quantity (default: None)
        length of slices of spike trains, needed if dimensionality=True
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, Lslice=None, dimensionality=False,
                            **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        if not hasattr(model, 'prediction'):
            model.prediction = {}
        if self.test_hash in model.prediction:
            evals = model.prediction[self.test_hash]
        else:
            spiketrains = model.produce_spiketrains(**self.params)
            if not self.params['dimensionality']:
                cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                                    **self.params)
                evals, _ = eigh(cc_matrix)
                model.prediction[self.test_hash] = evals
                result = evals
            elif self.params['Lslice'] is not None:
                t0 = spiketrains[0].t_start
                duration = spiketrains[0].t_stop - t0
                nt = int((duration/Lslice).rescale('').magnitude)
                if not nt > 0:
                    raise KeyError('Keyword "Lslice" not set correctly.'\
                                   +' Number of slices smaller than 1')
                dim = np.zeros(nt)
                for i in range(nt):
                    spktr = [st.time_slice(t0+i*Lslice, (i+1)*Lslice) \
                             for st in spiketrains]
                    cc_matrix = self.generate_cc_matrix(spiketrains=spktr,
                                                        **self.params)
                    cc_matrix[np.isnan(cc_matrix)] = 0
                    evals, _ = eigh(cc_matrix)
                    dim[i] = np.sum(evals)**2 / np.sum(evals**2)
                result = dim
        return result
