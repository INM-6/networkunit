# -*- coding: utf-8 -*-
#
# brunel_model.py
#
# This file derives from `brunel_exp_multisynapse_nest.py`
# It therefore derives from a file that is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Random balanced network (exp synapses, multiple time constants).

---------------------------------------------------------------

This script simulates an excitatory and an inhibitory population on
the basis of the network used in

Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
183-208 (2000).

The example demonstrate the usage of the multisynapse neuron
model. Each spike arriving at the neuron triggers an exponential
PSP. The time constant associated with the PSP is defined in the
recepter type array tau_syn of each neuron. The receptor types of all
connections are uniformally distributed, resulting in uniformally
distributed time constants of the PSPs.

When connecting the network customary synapse models are used, which
allow for querying the number of created synapses. Using spike
detectors the average firing rates of the neurons in the populations
are established. The building as well as the simulation time of the
network are recorded.

Edited: Robin Gutzen and Aitor Morales-Gregorio; Nov 2019:
    * Wrap simulation around NetworkUnit class (following RG's work
    on the microcircuit model)

"""

import nest
import os

from networkunit.models import nest_simulation

from model_params import net_dict
from sim_params import sim_dict


class brunel_model(nest_simulation):

    def __init__(self,
                 name='',
                 backend='Nest',
                 attrs=None,
                 model_params=None,
                 stimulus_params=None):

        # Inherit from parent class
        super(brunel_model, self).__init__(name=name,
                                           backend=backend,
                                           attrs=attrs,
                                           model_params=net_dict)

        # Set simulation parameters, inherited from grandparent class
        self.use_default_run_params()
        self.run_params.update(sim_dict)

        # Handle data destination, when saving to disk
        if self.run_params['to_file']:
            self.data_path = self.run_params['data_path']
            if nest.Rank() == 0:
                if os.path.isdir(self.data_path):
                    print(self.data_path + ' already exists')
                else:
                    os.mkdir(self.data_path)
                    print('Data directory created')
                print('Data will be written to {}'.format(self.data_path))
        pass

    def init_simulation(self):
        """
        Hand parameters to the NEST-kernel.

        Reset the NEST-kernel and pass parameters to it.

        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.
        """
        # super(brunel_model, self).init_simulation()

        nest.ResetKernel()

        # Fetch simulation parameters from dictionary and set the nest kernel
        kernel_params = {"resolution": self.run_params['resolution'],
                         "print_time": self.run_params['print_time'],
                         "overwrite_files": self.run_params['overwrite_files'],
                         'total_num_virtual_procs':
                             self.run_params['total_num_virtual_procs']}

        nest.SetKernelStatus(kernel_params)

        # print out the number of processes
        N_tp = nest.GetKernelStatus('total_num_virtual_procs')
        if nest.Rank() == 0:
            print('Number of total processes: {}'.format(N_tp))

        pass

    def setup_network(self):
        """
        Create neural populations and measurement devices.

        Parameters are inherited from itself and
        from the simulation dictionary.
        """

        # Configuration of synapse model and poisson generator
        nest.SetDefaults("iaf_psc_exp_multisynapse",
                         self.model_params['neuron_params'][0])
        nest.SetDefaults("poisson_generator",
                         {"rate": self.model_params['p_rate']})

        # Create populations
        self.nodes_ex = nest.Create("iaf_psc_exp_multisynapse",
                                    self.model_params['N_E'])
        self.nodes_in = nest.Create("iaf_psc_exp_multisynapse",
                                    self.model_params['N_I'])

        # Create Poisson generator
        self.noise = nest.Create("poisson_generator")

        # Create spike detectors
        self.espikes = nest.Create("spike_detector")
        self.ispikes = nest.Create("spike_detector")
        nest.SetStatus(self.espikes, [{"label": "brunel-py-ex",
                                       "withtime": True,
                                       "withgid": True,
                                       "to_file": self.run_params['to_file']}])
        nest.SetStatus(self.ispikes, [{"label": "brunel-py-in",
                                       "withtime": True,
                                       "withgid": True,
                                       "to_file": self.run_params['to_file']}])
        pass

    def connect_network(self):
        """
        Create connections between neurons and with devices.

        First set up synapse models, then create the connections.
        """

        # Set up synapse models:
        # Copy synapse models from NEST
        nest.CopyModel("static_synapse", "excitatory",
                       {"weight": self.model_params['J_ex'],
                        "delay": self.model_params['delay']})
        nest.CopyModel("static_synapse", "inhibitory",
                       {"weight": self.model_params['J_in'],
                        "delay": self.model_params['delay']})

        # Set synapse model params
        syn_ex = {"model": "excitatory",
                  "receptor_type": {"distribution": "uniform_int",
                                    "low": 1,  # XXX Is this a parameter?
                                    "high": self.model_params['nr_ports']
                                    }
                  }
        syn_in = {"model": "inhibitory",
                  "receptor_type": {"distribution": "uniform_int",
                                    "low": 1,  # XXX Is this a parameter?
                                    "high": self.model_params['nr_ports']
                                    }
                  }

        # Connect poisson generators
        nest.Connect(self.noise, self.nodes_ex,
                     syn_spec=syn_ex)
        nest.Connect(self.noise, self.nodes_in,
                     syn_spec=syn_ex)

        # Connect spike detectors
        nest.Connect(self.nodes_ex[:self.model_params['N_rec']],
                     self.espikes,
                     syn_spec="excitatory")
        nest.Connect(self.nodes_in[:self.model_params['N_rec']],
                     self.ispikes,
                     syn_spec="excitatory")

        # Connect neuron populations
        nest.Connect(self.nodes_ex,
                     self.nodes_ex,
                     self.model_params['conn_params_ee'],
                     syn_ex)
        nest.Connect(self.nodes_ex,
                     self.nodes_in,
                     self.model_params['conn_params_ie'],
                     syn_ex)
        nest.Connect(self.nodes_in,
                     self.nodes_in,
                     self.model_params['conn_params_ii'],
                     syn_in)
        nest.Connect(self.nodes_in,
                     self.nodes_ex,
                     self.model_params['conn_params_ei'],
                     syn_in)
        pass

    def get_status(self):
        """Unknown functionality"""
        # print network summary?
        pass
