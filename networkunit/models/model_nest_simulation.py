import sciunit
import quantities as pq
from .backends import available_backends

class nest_simulaton(sciunit.models.RunnableModel):
    # ToDo: How to use attrs?

    def __init__(name, backend='Nest', attrs=None, model_params=None):
        super(nest_simulaton, self).__init__(name=name,
                                             backend=backend,
                                             attrs=attrs)

        if not hasattr(self, model_params):
            self.model_params = {}
        if model_params is not None:
            self.model_params.update(model_params)


        # ToDo: define general defaults, use Nest's own defaults when given
        default_run_params = {"resolution"     : 1*pq.ms,
                              "print_time"     : True,
                              "overwrite_files": True,
                              "grng_seed"      : 0000,
                              "rng_seeds"      : [seed*42],
                              "simtime"        : 1000*pq.ms}

        set_default_run_params(self, **default_run_params)
        return None


    def init_simulation(self):
        """Initializes the Nest simulation with the run_params.
        Is called from self.backend._backend_run()."""

        nest.ResetKernel()
        # ToDo: add all possible settings, and define defaults
        nest.SetKernelStatus({"resolution"     : self.run_params['resolution'],
                              "print_time"     : self.run_params['print_time'],
                              "overwrite_files": self.run_params['overwrite_files'],
                              "grng_seed"      : self.run_params['grng_seed'],
                              "rng_seeds"      : self.run_params['rng_seeds']})
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
