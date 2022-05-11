from networkunit.tests.two_sample_test import two_sample_test
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import isi, lv, cv, cv2, lvr
from networkunit.utils import use_cache, filter_valid_params, parallelize
import numpy as np


class isi_variation_test(two_sample_test):
    """
    Test to compare the firing rates of a set of spiking neurons in a network.

    Parameters:
    ----------
    variation_measure: 'isi', 'cv', 'lv', 'lvr' (default)
        'isi' - Compares the inter-spike intervals
        'cv'  - Compares the coefficients of variation
        'cv2'  - Compares the coefficients of variation
        'lv'  - Compares the local coefficients of variation
        'lvr'  - Compares the revised local coefficients of variation
    """

    required_capabilities = (ProducesSpikeTrains, )
    default_params = {**two_sample_test.default_params,
                      'variation_measure': 'lvr',
                      'with_nan': True}

    @use_cache
    def generate_prediction(self, model):
        if self.params['variation_measure'] not in ['isi', 'cv', 'cv2', 'lv', 'lvr']:
            raise ValueError('Variation measure not known.')

        spiketrains = model.produce_spiketrains(**self.params)

        with parallelize(isi, self) as isi_parallel:
            intervals = isi_parallel(spiketrains)

        if self.params['variation_measure'] == 'isi':
            return np.concatenate(intervals) * intervals.units

        var_measure_func = globals()[self.params['variation_measure']]

        with filter_valid_params(var_measure_func) as _var_measure:
            with parallelize(_var_measure, self) as var_measure_parallel:
                isi_var = var_measure_parallel(intervals, **self.params)

        return np.array(isi_var)
