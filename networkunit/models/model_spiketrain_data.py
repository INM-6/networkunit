from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.models.model_loaded_data import loaded_data
from networkunit.plots.plot_rasterplot import rasterplot
from neo.core import SpikeTrain
from neo.io import NeoHdf5IO
from copy import copy
import numpy as np
import os
import neo


class spiketrain_data(loaded_data, ProducesSpikeTrains):
    """
    Abstract model class for spiking data.
    It has an example loading routine for hdf files, is able to display the
    corresponding rasterplot with self.show_rasterplot(),
    and if the self.params contains:
    align_to_0=True, the spiketrains all start from 0s,
    max_subsamplesize=x, only the x first spike trains are used.
    """
    def load(self, file_path=None, client=None, **kwargs):
        """
        Loads spiketrains from a hdf5 file in the neo data format.

        Parameters
        ----------
        file_path : string
            Path to file
        client :
            When file is loaded from a collab storage a appropriate client
            must be provided.
        Returns :
            List of neo.SpikeTrains
         """
        if file_path is None:
            file_path = self.file_path
        if file_path[-2:] != 'h5':
            raise IOError('file must be in hdf5 file in Neo format')

        if client is None:
            data = NeoHdf5IO(file_path)
        else:
            store_path = './' + file_path.split('/')[-1]
            client.download_file(file_path, store_path)
            data = NeoHdf5IO(store_path)

        spiketrains = data.read_block().list_children_by_class(SpikeTrain)

        for i in xrange(len(spiketrains)):
            spiketrains[i] = spiketrains[i].rescale('ms')

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
        self.params.update(kwargs)
        self.spiketrains = self.load(file_path=self.file_path, **self.params)
        if type(self.spiketrains) == list:
            for st in self.spiketrains:
                if type(st) == neo.core.spiketrain.SpikeTrain:
                    pass
        else:
            raise TypeError('loaded data is not a list of neo.SpikeTrain')

        self.spiketrains = self.preprocess(self.spiketrains, **self.params)
        return self.spiketrains

    def show_rasterplot(self, **kwargs):
        return rasterplot(self.spiketrains, **kwargs)
