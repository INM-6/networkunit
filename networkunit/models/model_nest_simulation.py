import sciunit
from .backends import available_backends
try:
    import nest
    nest_available = True
except ImportError:
    nest_available = False
    nest = None


class nest_simulation(sciunit.models.RunnableModel):
    # ToDo: How to use attrs?

    def __init__(self, name,
                 backend='Nest',
                 attrs=None,
                 model_params=None):

        super(nest_simulation, self).__init__(name=name,
                                              backend=backend,
                                              attrs=attrs)

        if not hasattr(self, 'model_params'):
            self.model_params = {}
        if model_params is not None:
            self.model_params.update(model_params)

        # # Default kernel status `nest.GetKernelStatus()`
        # nest.ResetKernel()
        # default_run_params = nest.GetKernelStatus()
        #
        # # Remove keys that could raise an error while setting them
        # non_settable_params = ['T_max',
        #                        'T_min',
        #                        'tics_per_step',
        #                        'time_collocate',
        #                        'time_communicate',
        #                        'to_do',
        #                        'min_delay',
        #                        'max_delay']
        # for key in non_settable_params:
        #     try:
        #         default_run_params.pop(key)
        #     except KeyError:
        #         print(key + ' was not in the default_run_params dict')
        #
        # self.set_default_run_params(**default_run_params)
        return None

    def init_simulation(self):
        """
        Initialize the Nest simulation with the run_params.

        Is called from self.backend._backend_run().
        """
        kernel_params = self.default_run_params
        kernel_params.update(self.run_params)
        nest.ResetKernel()
        nest.SetKernelStatus(kernel_params)
        pass

    def init_model(self):
        """
        Set up and connect the network model with model_params.

        Is called from self.backend._backend_run().
        """

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
