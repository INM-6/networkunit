import numpy as np
from networkunit.tests.two_sample_test import two_sample_test
from networkunit.utils import generate_prediction_wrapper

class joint_test(two_sample_test):
    """
    Test class which enables the joint evaluation of features from other tests.
    It creates the predictions of the tests specified in 'test_list' with the
    parameters from 'test_params', and combines them in a MxN float array, where
    M is the number of tests and N the number of prediction values
    (needs to be equal for all tests).
    The joint test can only be paired with a score_type which can compare
    multidimensional distributions, e.g. kl_divergence, wasserstein_distance.

    Example:
    ```
    class fr_lv_jtest(tests.TestM2M, tests.joint_test):
        score_type = scores.kl_divergence
        params = {}
        test_list = [tests.firing_rate_test,
                     tests.isi_variation_test,
                     tests.isi_variation_test]
        test_params = [{},
                       {'variation_measure': 'lv'},
                       {'variation_measure': 'cv'}]
    ```
    """

    def check_tests(self, model):
        if not hasattr(self, 'test_list') or not isinstance(self.test_list, list):
            raise AttributeError("Joint test doesn't define a test_list!")
        if not hasattr(self, 'test_params') or not isinstance(self.test_params, list):
            raise AttributeError("Joint test doesn't define a test_params list!")
        if len(self.test_list) - len(self.test_params):
            raise AttributeError("test_list and test_params are not of same length!")
        for test, params in zip(self.test_list, self.test_params):
            if not isinstance(test, type):
                raise TypeError("{} not a legit test class!".format(test))
            if not isinstance(params, dict):
                raise TypeError("{} doesn't have legit params dict!".format(test))

        sts = model.produce_spiketrains()
        grouped_sts = model.produce_grouped_spiketrains()
        flat_sts = [st for st_list in grouped_sts for st in st_list]
        for st1, st2 in zip(sts, flat_sts):
            if not np.allclose(st1.times, st2.times):
                raise ValueError('flattened grouped spiketrains and spiketrains must have the same ordering!')
        del grouped_sts
        del flat_sts
        del sts
        pass

    @generate_prediction_wrapper
    def generate_prediction(self, model, **params):
        self.check_tests(model)

        prediction = []
        self.test_inst = []

        for test_class, test_params in zip(self.test_list, self.test_params):
            if 'name' in test_params.keys():
                test_name = test_params.pop('name')
            else:
                test_name = None
            self.test_inst.append(
                test_class(observation=self.observation,
                           name=test_name,
                           **test_params))

        # ToDO: consider parallelization
        for test in self.test_inst:
            pred = np.array(test.generate_prediction(model))
            if len(pred.shape) > 1:
                for i in range(pred.shape[-1]):
                    prediction.append(pred[:, i])
            else:
                prediction.append(pred)

        it = iter(prediction)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('Not all predictions have the same length!')
        prediction = np.array(prediction, dtype=float)

        return prediction
