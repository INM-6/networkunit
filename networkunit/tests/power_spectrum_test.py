from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from elephant.signal_processing import zscore
from networkunit.plots.power_spectral_density import power_spectral_density
import numpy as np
import quantities as pq
from inspect import signature

class power_spectrum_test(two_sample_test):
    """
    Test to compare the power spectral density of a set of spiking neurons
    in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.
    Parameters are passed on to elephant.spectral.welch_psd
    """

    required_capabilities = (ProducesSpikeTrains,)

    name = 'Power-Spectrum Test'
    default_params = {'freq_res': 1,
                      'binszie': 5*pq.ms,
                      'psd_precision': 0.0001
                      }

    def generate_prediction(self, model, **kwargs):
        psd_samples = self.get_prediction(model)
        if psd_samples is None:
            if kwargs:
                self.params.update(kwargs)

            # calculating population activity
            spiketrains = model.produce_spiketrains(**self.params)
            asignal = time_histogram(spiketrains,
                                     binsize=self.params['binsize'])
            zscore(asignal, inplace=True)

            # calculating power spectrum with params
            psd_sig = signature(welch_psd)[0]
            psd_params = dict((k, self.params[k]) for k in
                              psd_sig.parameters.keys() if k in self.params)
            freqs, psd = welch_psd(asignal, **psd_params)
            psd /= (np.mean(psd)*len(freqs))
            breakpoint()

            # PSD values -> PSD samples
            factor = 1/self.params['psd_precision']
            psd_samples = np.array(([f]*int(np.round(p*factor))
                                    for f,p in zip(freqs, psd))).flatten()

            self.set_prediction(model, psd_samples)
        return psd_samples
