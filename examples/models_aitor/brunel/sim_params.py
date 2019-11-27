# -*- coding: utf-8 -*-
'''
Brunel network simulation parameters
----------------------------------

Simulation parameters for a Brunel-type network

Aitor Morales-Gregorio; Nov 2019
'''
import os

sim_dict = {
    # Simulation time (in ms).
    'simtime': 100.0,
    # Resolution of the simulation (in ms).
    'resolution': 0.1,
    # Path to save the output data.
    'data_path': os.path.join(os.getcwd(), 'brunel_simulation_devices/'),
    # Number of virtual processes
    # will be distributed between processes and threads
    'total_num_virtual_procs': 1,
    # Determine if simulation output files will be created
    'to_file': False,
    # If True, data will be overwritten,
    # If False, a NESTError is raised if the files already exist.
    'overwrite_files': True,
    # Print the time progress, this should only be used when the simulation
    # is run on a local machine.
    'print_time': False
    }
