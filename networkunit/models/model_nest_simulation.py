import sciunit
import quantities as pq
from .backends import available_backends
try:
    import nest
    nest_available = True
except ImportError:
    nest_available = False
    nest = None


class nest_simulation(sciunit.models.RunnableModel):
    # ToDo: How to use attrs?

    def __init__(self, name, backend='Nest', attrs=None, model_params=None):
        super(nest_simulation, self).__init__(name=name,
                                              backend=backend,
                                              attrs=attrs)

        if not hasattr(self, model_params):
            self.model_params = {}
        if model_params is not None:
            self.model_params.update(model_params)

        # Default kernel status `nest.GetKernelStatus()` on NEST 2.18.0
        default_run_params = {'adaptive_spike_buffers': True,
                              'adaptive_target_buffers': True,
                              'buffer_size_secondary_events': 0,
                              'buffer_size_spike_data': 2,
                              'buffer_size_target_data': 2,
                              'data_path': '',
                              'data_prefix': '',
                              'dict_miss_is_error': True,
                              'grng_seed': 0,
                              'growth_factor_buffer_spike_data': 1.5,
                              'growth_factor_buffer_target_data': 1.5,
                              'keep_source_table': True,
                              'local_num_threads': 1,
                              'local_spike_counter': 0,
                              'max_buffer_size_spike_data': 8388608,
                              'max_buffer_size_target_data': 16777216,
                              'max_num_syn_models': 512,
                              'max_delay': 0.1,
                              'min_delay': 0.1,
                              'ms_per_tic': 0.001,
                              'network_size': 1,
                              'num_connections': 0,
                              'num_processes': 1,
                              'off_grid_spiking': False,
                              'overwrite_files': False,
                              'print_time': False,
                              'resolution': 0.1,
                              'rng_seeds': (1,),
                              'sort_connections_by_source': True,
                              'structural_plasticity_synapses': {},
                              'structural_plasticity_update_interval': 1000,
                              'T_max': 1152921504606846.8,
                              'T_min': -1152921504606846.8,
                              'tics_per_ms': 1000.0,
                              'tics_per_step': 100,
                              'time': 0.0,
                              'time_collocate': 0.0,
                              'time_communicate': 0.0,
                              'to_do': 0,
                              'total_num_virtual_procs': 1,
                              'use_wfr': True,
                              'wfr_comm_interval': 1.0,
                              'wfr_interpolation_order': 3,
                              'wfr_max_iterations': 15,
                              'wfr_tol': 0.0001}

        self.set_default_run_params(**default_run_params)
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
