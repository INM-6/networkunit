import sciunit
import numpy as np
from networkunit.tests.test_two_sample_test import two_sample_test
from abc import ABCMeta, abstractmethod


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

    @property
    @abstractmethod
    def test_list(self):
        """list of test classes to combine"""
        pass

    @property
    @abstractmethod
    def test_params(self):
        """parameters (list of dicts) to be passed to the tests"""
        pass

    def check_tests(self):
        if not hasattr(self, 'test_list') or not isinstance(self.test_list, list):
            raise AttributeError("Joint test doesn't define a test_list!")
<<<<<<< HEAD
        if not hasattr(self, 'test_params') or not isinstance(self.test_params, list):
                raise AttributeError("Joint test doesn't define a test_params list!")
=======
        if not hasattr(self, 'test_params') or isinstance(self.test_params, list):
            raise AttributeError("Joint test doesn't define a test_params list!")
>>>>>>> d45a471eadde31a275ffa953714b39e1aa6a6b97
        if len(self.test_list) - len(self.test_params):
            raise AttributeError("test_list and test_params are not of same length!")
        for test, params in zip(self.test_list, self.test_params):
            if not isinstance(test, type):
                raise TypeError("{} not a legit test class!".format(test))
            if not isinstance(params, dict):
                raise TypeError("{} doesn't have legit params dict!".format(test))
        pass

    def generate_prediction(self, model, **kwargs):
        self.check_tests()
        if kwargs:
            self.params.update(kwargs)
        prediction = self.get_prediction(model)
        if prediction is None:
            prediction = []
            self.test_inst = []

            for test_class, params in zip(self.test_list, self.test_params):
                self.params.update(params)
                self.test_inst.append(test_class(observation=self.observation,
                                                 name=self.name, **self.params))

            for test in self.test_inst:
                prediction.append(test.generate_prediction(model))

            it = iter(prediction)
            the_len = len(next(it))
            if not all(len(l) == the_len for l in it):
                raise ValueError('Not all predictions have the same length!')
            prediction = np.array(prediction, dtype=float)

            self.set_prediction(model, prediction)
        return prediction
