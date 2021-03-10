from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
# from networkunit.plots.power_spectral_density import power_spectral_density
import numpy as np
import quantities as pq
from inspect import signature


class power_spectrum_test(two_sample_test):
    """
    Test to compare the power spectral density of a set of spiking neurons
    in a network.
    All spiketrains need to have the same t_start and t_stop.
    Parameters are passed on to elephant.spectral.welch_psd()
    """

    required_capabilities = (ProducesSpikeTrains,)

    default_params = {'frequency_resolution': 2.5,
                      'binsize': 10*pq.ms,
                      'psd_precision': 0.0001
                      }

    def generate_prediction(self, model, **kwargs):
        psd_samples = self.get_prediction(model)
        if psd_samples is None:
            if kwargs:
                self.params.update(kwargs)

            spiketrains_list = model.produce_grouped_spiketrains(**self.params)
            psd_samples = []

            for spiketrains in spiketrains_list:
                psd = self.spiketrains_psd(spiketrains)
                psd_samples.append(self.psd_to_samples(freqs, psd))

            self.set_prediction(model, psd_samples)
        return psd_samples


    def spiketrains_psd(self, spiketrains):
        if not (type(spiketrains) == list) \
        or not type(spiketrains[0]) == neo.SpikeTrain:
            raise TypeError("Input must be a list of neo.Spiketrain obejects.")

        asignal = time_histogram(spiketrains,
                                 binsize=self.params['binsize'])
        zscore(asignal, inplace=True)

        psd_sig = signature(welch_psd)[0]
        psd_params = dict((k, self.params[k]) for k in
                          psd_sig.parameters.keys() if k in self.params)
        freqs, psd = welch_psd(asignal, **psd_params)
        # Adjust so the integral over f is 1
        psd /= (np.mean(psd)*self.params['frequency_resolution'])
        return psd


    def psd_to_samples(self, freqs, psd):
        factor = 1/self.params['psd_precision']
        psd_factors = np.round(psd*factor).astype(int)
        return np.repeat(freqs, psd_factors)
