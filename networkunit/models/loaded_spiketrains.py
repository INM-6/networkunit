from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
from sciunit.models import RunnableModel
from networkunit.plots.rasterplot import rasterplot
from neo.core import SpikeTrain
from copy import copy
import numpy as np
import neo


class loaded_spiketrains(RunnableModel, ProducesSpikeTrains):
    """
    Abstract model class for spiking data.
    It has an example loading routine for hdf files, is able to display the
    corresponding rasterplot with self.show_rasterplot(),
    and if the self.params contains:
    align_to_0=True, the spiketrains all start from 0s,
    max_subsamplesize=x, only the x first spike trains are used.
    """

    default_params = {'file_path':None}

    def __init__(self, name=None, backend='storage', attrs=None, **params):
        """
        Parameters
        ----------
        name : string
            Name of model instance
        **params :
            class attributes to be stored in self.params
        """

        if not hasattr(self, 'default_params'):
            self.default_params = {}
        if not hasattr(self, 'params') or self.params is None:
            self.params = {}
        params = {**self.default_params, **self.params, **params}

        super(loaded_spiketrains, self).__init__(name=name,
                                                 backend=backend,
                                                 attrs=attrs,
                                                 **params)


    def load(self):
        """
        Loads spiketrains from a .nix file in the neo data format.

        Returns :
            List of neo.SpikeTrains
         """
        file_path = self.params['file_path']
        if file_path is None:
            raise ValueError('"file_path" parameter is not set!')
        if not file_path.endswith('.nix'):
            raise IOError('file must be in .NIX format')

        if client is None:
            with neo.NixIO(file_path) as nio:
                block = nio.read_block()
        else:
            store_path = './' + file_path.split('/')[-1]
            client.download_file(file_path, store_path)
            with neo.NixIO(store_path) as nio:
                block = nio.read_block()

        spiketrains = block.list_children_by_class(SpikeTrain)
        return spiketrains


    def _align_to_zero(self, spiketrains=None):
        if spiketrains is None:
            spiketrains = self.spiketrains
        t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
        tmin = min(t_lims, key=lambda f: f[0])[0]
        tmax = max(t_lims, key=lambda f: f[1])[1]
        unit = spiketrains[0].units
        for count, spiketrain in enumerate(spiketrains):
            annotations = spiketrain.annotations
            spiketrains[count] = SpikeTrain(
                np.array(spiketrain.tolist()) * unit - tmin,
                t_start=0 * unit,
                t_stop=tmax - tmin)
            spiketrains[count].annotations = annotations
        return spiketrains


    def preprocess(self, spiketrain_list, max_subsamplesize=None,
                   align_to_0=True, **kwargs):
        """
        Performs preprocessing on the spiketrain data according to the given
        parameters which are passed down from the test parameters.
        """
        if spiketrain_list is not None and max_subsamplesize is not None:
            spiketrains = spiketrain_list[:max_subsamplesize]
        else:
            spiketrains = copy(spiketrain_list)

        if align_to_0:
            spiketrains = self._align_to_zero(spiketrains)
        return spiketrains


    def produce_spiketrains(self, **kwargs):
        """
        overwrites function in capability class ProduceSpiketrains
        """
        self.spiketrains = self._backend.backend_run()
        if type(self.spiketrains) == list:
            for st in self.spiketrains:
                if type(st) == neo.core.spiketrain.SpikeTrain:
                    pass
        else:
            raise TypeError('loaded data is not a list of neo.SpikeTrain')

        self.params.update(kwargs)
        self.spiketrains = self.preprocess(self.spiketrains, **self.params)
        return self.spiketrains


    def show_rasterplot(self, **kwargs):
        return rasterplot(self.spiketrains, **kwargs)
