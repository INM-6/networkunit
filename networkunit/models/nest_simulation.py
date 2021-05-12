import sciunit
import quantities as pq
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains


class nest_simulation(sciunit.models.RunnableModel, ProducesSpikeTrains):
    # ToDo: How to use attrs?
    nest_instance = None

    def __init__(self, name, nest_instance=None, attrs=None, model_params=None):
        if nest_instance is None:
            import nest
            self.nest_instance = nest
        else:
            self.nest_instance = nest_instance

        super(nest_simulation, self).__init__(name=name,
                                              backend='Nest',
                                              attrs=attrs)

        self._backend.nest_instance = self.nest_instance

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
        self.nest_instance.ResetKernel()
        kernel_params = self.nest_instance.GetKernelStatus()
        kernel_params.update(self.run_params)
        self.nest_instance.SetKernelStatus(kernel_params)
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
