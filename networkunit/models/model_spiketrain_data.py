import sciunit
#from networkunit import capabilities as cap
#from networkunit import models

from networkunit.capabilities import ProducesSpikeTrains
from networkunit.models import simulation_data
from neo.core import SpikeTrain
from neo.io import NeoHdf5IO
from copy import copy
import numpy as np
import os
import neo

class spiketrain_data(simulation_data, ProducesSpikeTrains):
    """
    A model class to wrap network activity data (in form of spike trains) from
    an already performed simulation of the Potjans-Diesman cortical
    microcircuit model.
    """
    def load(self, file_path, client=None, **kwargs):
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
            List of neo.SpikeTrains of length N
            """
        # Load NEST or SpiNNaker data using NeoHdf5IO
        if file_path[-2:] != 'h5':
            raise IOError, 'file must be in hdf5 file in Neo format'

        if client is None:
            data = NeoHdf5IO(file_path)
        else:
            store_path = './' + file_path.split('/')[-1]
            client.download_file(file_path, store_path)
            data = NeoHdf5IO(store_path)

        spiketrains = data.read_block().list_children_by_class(SpikeTrain)
        return spiketrains

    def _align_to_zero(self, spiketrains=None):
        if spiketrains is None:
            spiketrains = self.spiketrains
        t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
        tmin = min(t_lims, key=lambda f: f[0])[0]
        tmax = max(t_lims, key=lambda f: f[1])[1]
        unit = spiketrains[0].units
        for count, spiketrain in enumerate(spiketrains):
            spiketrains[count] = SpikeTrain(
                np.array(spiketrain.tolist()) * unit - tmin,
                t_start=0 * unit,
                t_stop=tmax - tmin)
        return spiketrains

    def preprocess(self, spiketrain_list, max_subsamplesize=None,
                   align_to_0=True, **kwargs):
        """
        Performs preprocessing on the spiketrain data according to the given
        parameters which are passed down from the test test parameters.
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
        overwrites function in class ProduceCovariances
        """
        self.params.update(kwargs)
        self.spiketrains = self.data
        if type(self.spiketrains) == list:
            for st in self.spiketrains:
                if type(st) == neo.core.spiketrain.SpikeTrain:
                    pass
        else:
            raise TypeError, 'loaded data is not a list of neo.SpikeTrain'

        self.spiketrains = self.preprocess(self.spiketrains, **self.params)
        return self.spiketrains
