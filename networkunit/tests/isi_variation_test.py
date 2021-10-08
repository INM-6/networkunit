from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import isi, lv, cv2, lvr
from networkunit.utils import use_prediction_cache, filter_valid_params


class isi_variation_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.

    Parameters:
    ----------
    variation_measure: 'isi', 'cv', 'lv', 'lvr' (default)
        'isi' - Compares the inter-spike intervals
        'cv'  - Compares the coefficients of variation
        'lv'  - Compares the local coefficients of variation
        'lvr'  - Compares the revised local coefficients of variation
    """

    required_capabilities = (ProducesSpikeTrains, )
    default_params = {'variation_measure': 'lvr',
                      'with_nan': True}

    @use_prediction_cache
    def generate_prediction(self, model):
        spiketrains = model.produce_spiketrains(**self.params)
        isi_list = [isi(st) for st in spiketrains]
        if self.params['variation_measure'] == 'lv':
            isi_var = []
            with filter_valid_params(lv) as _lv:
                for intervals in isi_list:
                    isi_var.append(_lv(intervals, **self.params))
        elif self.params['variation_measure'] == 'cv':
            isi_var = []
            with filter_valid_params(cv2) as _cv2:
                for intervals in isi_list:
                    isi_var.append(_cv2(intervals, **self.params))
        elif self.params['variation_measure'] == 'isi':
            isi_var = [float(item) for sublist in isi_list
                       for item in sublist]
        elif self.params['variation_measure'] == 'lvr':
            isi_var = []
            with filter_valid_params(lvr) as _lvr:
                for intervals in isi_list:
                    isi_var.append(_lvr(intervals, **self.params))
        else:
            raise ValueError('Variation measure not known.')

        return isi_var
