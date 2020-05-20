from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import isi, lv, cv2
import numpy as np

class isi_variation_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.

    Parameters (in dict params)
    ----------
    variation_measure: 'isi', 'cv', 'lv' (default)
        'isi' - Compares the inter-spike intervals
        'cv'  - Compares the coefficients of variation
        'lv'  - Compares the local coefficients of variation
    """

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, with_nan=True, **kwargs):
        isi_var = self.get_prediction(model)
        if isi_var is None:
            if kwargs:
                self.params.update(kwargs)
            if 'variation_measure' not in self.params:
                self.params.update(variation_measure='lv')
            spiketrains = model.produce_spiketrains(**self.params)
            isi_list = [isi(st) for st in spiketrains]
            if self.params['variation_measure'] == 'lv':
                isi_var = []
                for intervals in isi_list:
                    isi_var.append(lv(np.squeeze(intervals), with_nan=with_nan))
            elif self.params['variation_measure'] == 'cv':
                isi_var = []
                for intervals in isi_list:
                    isi_var.append(cv2(np.squeeze(intervals), with_nan=with_nan))
            elif self.params['variation_measure'] == 'isi':
                isi_var = [float(item) for sublist in isi_list
                           for item in sublist]
            else:
                raise ValueError('Variation measure not known.')
            self.set_prediction(model, isi_var)
        return isi_var
