from networkunit.tests.test_two_sample_test import two_sample_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from elephant.statistics import isi, lv, cv #, cv2


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

    def generate_prediction(self, model, **kwargs):
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
                    if intervals.size > 2:
                        isi_var.append(lv(intervals))
            elif self.params['variation_measure'] == 'cv':
                isi_var = []
                for intervals in isi_list:
                    if intervals.size > 2:
                        isi_var.append(cv(intervals))
            elif self.params['variation_measure'] == 'isi':
                isi_var = [float(item) for sublist in isi_list for item in sublist]
            else:
                raise ValueError('Variation measure not known.')
            self.set_prediction(model, isi_var)
        return isi_var
