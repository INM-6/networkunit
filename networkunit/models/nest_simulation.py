import sciunit
import quantities as pq
from .backends import available_backends
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains
import sys
try:
    import nest
    nest_available = True
except ImportError:
    nest_available = False
    nest = None


class nest_simulation(sciunit.models.RunnableModel, ProducesSpikeTrains):
    # ToDo: How to use attrs?
    def __init__(self, name, backend='Nest', attrs=None, model_params=None):
        print('in nest simulation')
        super(nest_simulation, self).__init__(name=name,
                                              backend=backend,
                                              attrs=attrs)

        if not hasattr(self, 'model_params'):
            self.model_params = {}
        if model_params is not None:
            self.model_params.update(model_params)

        default_run_params = {"resolution": 1*pq.ms,
                              "print_time": True,
                              "overwrite_files": True,
                              "simtime": 1000*pq.ms}

        self.set_default_run_params(**default_run_params)
        return None

    def init_simulation(self):
        """Initializes the Nest simulation with the run_params.
        Is called from self.backend._backend_run()."""
        nest.ResetKernel()
        kernel_params = nest.GetKernelStatus()
        kernel_params.update(self.run_params)
        nest.SetKernelStatus(kernel_params)
        return None

    def init_model(self):
        """Setups and connects the network model with model_params.
        Is called from self.backend._backend_run()."""

        # setup network
        self.setup_network()

        # connect network
        self.connect_network()

        # get status
        self.get_status()
        return None

    def setup_network(self):
        """
        Set up the network.
        SetDefaults, Create nodes, SetStatus, ...
        using self.model_params
        """
        raise NotImplementedError("")

    def connect_network(self):
        """
        Connect the nodes and devices.
        CopyModel, Connect, ...
        using self.model_params
        """
        raise NotImplementedError("")

    def get_status(self):
        """
        Return and print the properties of the build network.
        GetConnections, GetStatus, ...
        using self.model_params
        """
        raise NotImplementedError("")

    def check_run_params(self):
        """Check if the parameters are appropriate for the model"""
        if False:
            raise sciunit.BadParameterValueError
        pass

    # def produce_spiketrains(self):
        # transform self.results (=nest.GetStatus) to list of neo.Spiketrains
