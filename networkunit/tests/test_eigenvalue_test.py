from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from abc import ABCMeta, abstractmethod
from scipy.linalg import eigh


class eigenvalue_test(correlation_test):

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        if not hasattr(model, 'prediction'):
            model.prediction = {}
        if self.test_hash in model.prediction:
            ews = model.prediction[self.test_hash]
        else:
            spiketrains = model.produce_spiketrains(**self.params)
            cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                                    **self.params)
            ews, _  = eigh(cc_matrix)
            model.prediction[self.test_hash] = ews
        return ews


