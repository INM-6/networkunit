from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import time_histogram
from elephant.spectral import welch_psd
from networkunit.plots.plot_power_spectral_density import power_spectral_density
import numpy as np

from elephant.statistics import time_histogram
from elephant.spectral import welch_psd


class power_spectrum_test(two_sample_test):
    """
    Test to compare the power spectral density of a set of spiking neurons in a network.
    The statistical testing method needs to be set in form of a
    sciunit.Score as score_type.
    Parameters are passed on to elephant.spectral.welch_psd
    """

    required_capabilities = (ProducesSpikeTrains,)

    def generate_prediction(self, model, **kwargs):
        psd = self.get_prediction(model)
        if psd is None:
            if kwargs:
                self.params.update(kwargs)

            spiketrains = model.produce_spiketrains(**self.params)

            self._set_default_param('binsze', 10 * pq.ms)
            self._set_default_param('num_seg', None)
            self._set_default_param('len_seg', None)
            self._set_default_param('freq_res', 1.)
            self._set_default_param('overlap', 0.5)
            self._set_default_param('fs', 100)
            self._set_default_param('window', 'hanning')
            self._set_default_param('nfft', None)
            self._set_default_param('detrend', 'constant')
            self._set_default_param('return_onesided', True)
            self._set_default_param('scaling', 'density')
            self._set_default_param('axis', -1)

            asignal = time_histogram(spiketrains, binsize=self.params['binsize'])
            freqs, psd = welch_psd(asignal,
                                   num_seg=self.params['num_seg'],
                                   len_seg=self.params['len_seg'],
                                   freq_res=self.params['freq_res'],
                                   overlap=self.params['overlap'],
                                   fs=self.params['fs'],
                                   window=self.params['window'],
                                   nfft=self.params['nfft'],
                                   detrend=self.params['detrend'],
                                   return_onesided=self.params['return_onesided'],
                                   scaling=self.params['scaling'],
                                   axis=self.params['axis'])
            model.psd_freqs = freqs
            # ToDo: How to quantitatively compare PSD distributions ??
            psd = np.squeeze(psd)
            self.set_prediction(model, psd)
        return psd

    def _set_default_param(self, pname, value):
        if pname not in self.params:
            self.params[pname] = value
        return None

    def visualize_samples(self, model1=None, model2=None, ax=None,
                          palette=None, sample_names=['observation', 'prediction'],
                          **kwargs):

        samples, palette, names = self._create_plotting_samples(model1=model1,
                                                                model2=model2,
                                                                palette=palette)
        if self.observation is None:
            sample_names[0] = model1.name
            freqs1 = model1.psd_freqs
            freqs2 = None
            if model2 is not None:
                sample_names[1] = model2.name
                freqs2 = model2.psd_freqs
        else:
            sample_names[1] = model1.name
            freqs1 = model1.psd_freqs
            freqs2 = model1.psd_freqs

        if len(samples) == 1:
            sample_2 = None
        else:
            sample_2 = samples[1]

        power_spectral_density(sample1=samples[0], freqs1=freqs1,
                               sample2=sample_2, freqs2=freqs2, ax=ax,
                               palette=palette, sample_names=sample_names, **kwargs)
        return ax
