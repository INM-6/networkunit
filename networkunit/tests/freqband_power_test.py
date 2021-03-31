from networkunit.tests.power_spectrum_test import power_spectrum_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
# from networkunit.plots.power_spectral_density import power_spectral_density
import numpy as np
import quantities as pq
from inspect import signature


class freqband_power_test(power_spectrum_test):
    """
    Test to compare the power spectral density of a set of spiking neurons
    in a network.
    All spiketrains need to have the same t_start and t_stop.
    Parameters are passed on to elephant.spectral.welch_psd()
    """

    default_params = {'frequency_resolution': 2.5,
                      'binsize': 10*pq.ms,
                      'psd_precision': 0.0001,
                      'highpass_freq': 13*pq.Hz,
                      'lowpass_freq': 20*pq.Hz,
                      }

    def generate_prediction(self, model, **kwargs):
        psd_samples = self.get_prediction(model)
        if psd_samples is None:
            if kwargs:
                self.params.update(kwargs)

            spiketrains_list = model.produce_grouped_spiketrains(**self.params)
            band_powers = []

            for spiketrains in spiketrains_list:
                freqs, psd = self.spiketrains_psd(spiketrains)
                f0 = np.argmax(freqs >= self.params['highpass_freq'])
                f1 = np.argmax(freqs >= self.params['lowpass_freq'])
                band_powers.append(np.mean(psd[f0:f1]))

            self.set_prediction(model, band_powers)
        return band_powers
