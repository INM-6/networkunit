from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import isi, lv, cv2, lvr
from networkunit.utils import generate_prediction_wrapper


class isi_variation_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.

    Parameters (in dict params)
    ----------
    variation_measure: 'isi', 'cv', 'lv' (default)
        'isi' - Compares the inter-spike intervals
        'cv'  - Compares the coefficients of variation
        'lv'  - Compares the local coefficients of variation
        'lvr'  - Compares the revised local coefficients of variation
    """

    required_capabilities = (ProducesSpikeTrains, )
    default_params = {'variation_measure': 'lvr',
                      'with_nan': True}

    @generate_prediction_wrapper
    def generate_prediction(self, model, **params):
        spiketrains = model.produce_spiketrains(**self.params)
        isi_list = [isi(st) for st in spiketrains]
        measure = params.pop('variation_measure')
        if measure == 'lv':
            isi_var = []
            for intervals in isi_list:
                isi_var.append(lv(intervals, **params))
        elif measure == 'cv':
            isi_var = []
            for intervals in isi_list:
                isi_var.append(cv2(intervals, **params))
        elif measure == 'isi':
            isi_var = [float(item) for sublist in isi_list
                       for item in sublist]
        elif measure == 'lvr':
            isi_var = []
            for intervals in isi_list:
                isi_var.append(lvr(intervals, **params))
        else:
            raise ValueError('Variation measure not known.')

        return isi_var
