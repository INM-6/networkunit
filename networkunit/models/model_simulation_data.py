import sciunit


class simulation_data(sciunit.Model):
    """
    Abstract model class which initializes self.params and loads simulation
    via self.load() into self.data.
    Child class needs to define load() function and file_path.
    """
    @property
    def file_path(self):
        raise NotImplementedError

    def __init__(self, name=None, **params):
        """
        Parameters
        ----------
        name : string
            Name of model instance
        **params :
            class attributes to be stored in self.params
        """
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
