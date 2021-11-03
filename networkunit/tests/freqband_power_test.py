from networkunit.tests.power_spectrum_test import power_spectrum_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.utils import use_cache, parallelize
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
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

    default_params = {**power_spectrum_test.default_params,
                      'highpass_freq': 13*pq.Hz,
                      'lowpass_freq': 20*pq.Hz,
                      }

    @use_cache
    def generate_prediction(self, model):
        spiketrains_list = model.produce_grouped_spiketrains(**self.params)

        with parallelize(self.calculate_band_powers, self) as calc_powers:
            band_powers = calc_powers(spiketrains_list)

        return np.array(band_powers)


    def calculate_band_powers(self, spiketrains):
        freqs, psd = self.spiketrains_psd(spiketrains)
        f0 = np.argmax(freqs >= self.params['highpass_freq'])
        f1 = np.argmax(freqs >= self.params['lowpass_freq'])
        return(np.mean(psd[f0:f1]))
