import sciunit
from abc import ABCMeta, abstractmethod


class model2model_test(sciunit.Test):
    """
    to be replaced with sciunit.TestM2M
    """
    __metaclass__ = ABCMeta

    def __init__(self, observation, name=None, **params):
        """
        Parameters
        ----------
        observation : sciUnit.Model instance
        """
        if params is None:
            params = {}
        self.params.update(params)
        self.observation_params = observation.params
        self.observation_model = observation
        observation = self.generate_prediction(observation, **params)
        super(model2model_test, self).__init__(observation, name=name, **params)

    @abstractmethod
    def generate_prediction(self, model, **kwargs):
        raise NotImplementedError("")
