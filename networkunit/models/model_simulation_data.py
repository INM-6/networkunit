import sciunit
from abc import ABCMeta, abstractmethod, abstractproperty


class simulation_data(sciunit.Model):

    # __metaclass__ = ABCMeta

    @property
    def file_path(self):
        raise NotImplementedError

    def __init__(self, name=None, **params):
        self.data = self.load(file_path=self.file_path, **params)
        super(simulation_data, self).__init__(name=name, **params)

    def load(self, file_path, **kwargs):
        raise NotImplementedError
