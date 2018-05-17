import sciunit
from abc import ABCMeta, abstractmethod, abstractproperty


class simulation_data(sciunit.Model):

    @property
    def file_path(self):
        raise NotImplementedError

    def __init__(self, name=None, **params):
        if params is None:
            params = {}
        if hasattr(self, 'params'):
            self.params.update(params)
        else:
            self.params = params
        self.data = self.load(file_path=self.file_path, **self.params)
        super(simulation_data, self).__init__(name=name, **self.params)

    def load(self, file_path, **kwargs):
        raise NotImplementedError
