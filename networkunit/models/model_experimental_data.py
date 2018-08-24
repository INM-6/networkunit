import sciunit
import neo
from networkunit.plots.plot_rasterplot import rasterplot


class experimental_data(sciunit.Model):
    """
    Abstract model class which initializes self.params and loads experimental
    data via self.load() into self.data.
    Child class needs to define load function and file_path.
    """
    @property
    def datfile(self):
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
        self.data = self.load(self.datfile, **self.params)
        super(experimental_data, self).__init__(name=name, **self.params)

    def load(self, datfile, **kwargs):
        raise NotImplementedError

    def produce_spiketrains(self, **kwargs):
        """
        overwrites function in capability class ProduceSpiketrains
        """
        self.params.update(kwargs)
        self.spiketrains = self.data
        if type(self.spiketrains) == list:
            for st in self.spiketrains:
                if type(st) == neo.core.spiketrain.SpikeTrain:
                    pass
        else:
            raise TypeError('loaded data is not a list of neo.SpikeTrain')
            
        return self.spiketrains

    def show_rasterplot(self, **kwargs):
        return rasterplot(self.spiketrains, **kwargs)