import numpy as np
from networkunit.tests.two_sample_test import two_sample_test
from networkunit.utils import use_cache, parallelize
from copy import copy

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
        test_list = [tests.firing_rate_test,
                     tests.isi_variation_test,
                     tests.isi_variation_test]
        test_params = [{},
                       {'variation_measure': 'lv'},
                       {'variation_measure': 'cv'}]
    ```
    """

    default_params = {**two_sample_test.default_params}

    def check_tests(self, model):
        if not hasattr(self, 'test_list') or not isinstance(self.test_list, list):
            raise AttributeError("Joint test doesn't define a test_list!")
        if not hasattr(self, 'test_params') or not isinstance(self.test_params, list):
            raise AttributeError("Joint test doesn't define a test_params list!")
        if len(self.test_list) - len(self.test_params):
            raise AttributeError("test_list and test_params are not of same length!")
        for test, test_params in zip(self.test_list, self.test_params):
            if not isinstance(test, type):
                raise TypeError("{} not a legit test class!".format(test))
            if not isinstance(test_params, dict):
                raise TypeError("{} doesn't have legitimate test_params dict!".format(test))

        sts = model.produce_spiketrains()
        grouped_sts = model.produce_grouped_spiketrains()
        flat_sts = [st for st_list in grouped_sts for st in st_list]
        for st1, st2 in zip(sts, flat_sts):
            if not np.allclose(st1.times.magnitude, st2.times.magnitude):
                raise ValueError('flattened grouped spiketrains and spiketrains must have the same ordering!')
        del grouped_sts
        del flat_sts
        del sts
        pass

    @use_cache
    def generate_prediction(self, model):
        self.check_tests(model)

        self.test_inst = []

        for test_class, test_params in zip(self.test_list, self.test_params):
            # Params priority order:
            # test params > joint-test params > joint-test default params
            # > test default params
            self.test_inst.append(
                test_class(observation=self.observation,
                           **{**self.params, **test_params}))

        def generate_test_prediction(test_inst):
            return np.array(test_inst.generate_prediction(model))

        with parallelize(generate_test_prediction, test_class=self) as parallel_test_predictions:
            prediction = parallel_test_predictions(self.test_inst)

        it = iter(prediction)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('Not all predictions have the same length!')
        prediction = np.array(prediction, dtype=float)

        return prediction
