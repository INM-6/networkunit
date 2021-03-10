from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
# from networkunit.plots.power_spectral_density import power_spectral_density
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

    default_params = {'frequency_resolution': 2.5*pq.Hz,
                      'binsize': 10*pq.ms,
                      'psd_precision': 0.0001
                      }

    def generate_prediction(self, model, **kwargs):
        psd_samples = self.get_prediction(model)
        if psd_samples is None:
            if kwargs:
                self.params.update(kwargs)

            spiketrains_list = model.produce_grouped_spiketrains(**self.params)
            # psd_samples = []
            psd_lst = []

            for spiketrains in spiketrains_list:
                freqs, psd = self.spiketrains_psd(spiketrains)
                psd_lst.append(psd)
                # psd_samples.append(self.psd_to_samples(freqs, psd))

            psd_arr = np.stack(psd_lst, axis=-1)
            mean_psd = np.mean(psd_arr, axis=-1)
            # std_psd = np.std(psd_arr, axis=-1)

            psd_samples = self.psd_to_samples(freqs, mean_psd)

            self.set_prediction(model, psd_samples)
        return psd_samples

    def spiketrains_psd(self, spiketrains):
        if not (type(spiketrains) == list) \
          or not type(spiketrains[0]) == neo.SpikeTrain:
            raise TypeError("Input must be a list of neo.Spiketrain obejects.")

        asignal = time_histogram(spiketrains,
                                 binsize=self.params['binsize'])
        zscore(asignal, inplace=True)

        psd_sig = signature(welch_psd)
        psd_params = dict((k, self.params[k]) for k in
                          psd_sig.parameters.keys() if k in self.params)
        freqs, psd = welch_psd(asignal, **psd_params)
        # Enforce single dimension shape, since
        # asignal will always be one-dimensional in this case
        psd = psd[0, :].magnitude

        # If there are nonzero values in the psd, avoid division by 0
        if psd.any():
            # Adjust so the integral over f is 1
            psd /= (np.nanmean(psd)*self.params['frequency_resolution'])

        return freqs, psd

    def psd_to_samples(self, freqs, psd):
        factor = 1/self.params['psd_precision']
        psd_factors = np.round(psd*factor).astype(int)
        return np.repeat(freqs, psd_factors)
