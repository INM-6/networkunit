from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import use_cache, filter_valid_params, parallelize
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
import numpy as np
import quantities as pq
from inspect import signature
import neo

class power_spectrum_test(two_sample_test):
    """
    Test to compare the power spectral density of a set of spiking neurons
    in a network.
    All spiketrains need to have the same t_start and t_stop.
    Parameters are passed on to elephant.spectral.welch_psd()
    """

    required_capabilities = (ProducesSpikeTrains,)

    default_params = {**two_sample_test.default_params,
                      'frequency_resolution': 2.5*pq.Hz,
                      'bin_size': 10*pq.ms,
                      'psd_precision': 0.0001
                      }

    @use_cache
    def generate_prediction(self, model):
        spiketrains_list = model.produce_grouped_spiketrains(**self.params)

        with parallelize(self.spiketrains_psd, self) as psd_parallel:
            psd_out = psd_parallel(spiketrains_list)

        freqs = psd_out[0][0]
        psd_lst = [out[1] for out in psd_out]
        psd_arr = np.stack(psd_lst, axis=-1)
        mean_psd = np.mean(psd_arr, axis=-1)

        psd_samples = self.psd_to_samples(freqs, mean_psd,
                                          self.params['psd_precision'])

        return psd_samples


    def spiketrains_psd(self, spiketrains):
        if not (type(spiketrains) == list) \
          or not type(spiketrains[0]) == neo.SpikeTrain:
            raise TypeError("Input must be a list of neo.Spiketrain obejects.")

        asignal = time_histogram(spiketrains,
                                 bin_size=self.params['bin_size'])
        zscore(asignal, inplace=True)

        with filter_valid_params(welch_psd) as _welch_psd:
            freqs, psd = _welch_psd(asignal, **self.params)

        # Enforce single dimension shape, since
        # asignal will always be one-dimensional in this case
        psd = psd[0, :].magnitude

        # If there are nonzero values in the psd, avoid division by 0
        if psd.any():
            # Adjust so the integral over f is 1
            psd /= (np.nanmean(psd)*self.params['frequency_resolution'])

        return freqs, psd


    def psd_to_samples(self, freqs, psd, precision):
        factor = 1 / precision
        psd_factors = np.round(psd*factor).astype(int)
        return np.repeat(freqs, psd_factors)
