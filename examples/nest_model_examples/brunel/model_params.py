# -*- coding: utf-8 -*-

'''
Brunel network model parameters
------------------------------

Model parameters for a Brunel-type network.

Aitor Morales-Gregorio; Nov 2019:
'''

net_dict = {
    # Neuron model.
    'neuron_model': 'iaf_psc_exp_multisynapse',
    # Number of neurons in the different populations. The order of the
    # elements corresponds to the names of the variable 'populations'.
    'N_E': 1000,
    'N_I': 250,
    # Connection probabilities. The first index corresponds to the targets
    # and the second to the sources.
    'P_EE': 0.1,
    'P_EI': 0.1,
    'P_IE': 0.1,
    'P_II': 0.1,
    # synaptic delay in ms
    'delay': 1.5,
    # Ratio inhibitory weight/excitatory weight
    'g': 5.0,
    # External rate relative to threshold rate
    'eta': 2.0,
    # Time constant of membrane potential in ms
    'tauMem': 20.0,
    # membrane threshold potential in mV
    'theta': 20.0,
    # postsynaptic amplitude in mV
    'J': 0.1,
    # number of receptor types
    'nr_ports': 100
    }

net_dict['N_rec'] = net_dict['N_E'] + net_dict['N_I']

indegrees = {
            # number of exc-exc synapses per excitatory target neuron
            'CEE': int(net_dict['P_EE'] * net_dict['N_E']),
            # number of inh-exc synapses per excitatory target neuron
            'CEI': int(net_dict['P_EI'] * net_dict['N_I']),
            # number of exc-inh synapses per inhibitory target neuron
            'CIE': int(net_dict['P_IE'] * net_dict['N_E']),
            # number of inh-inh synapses per inhibitory target neuron
            'CII': int(net_dict['P_II'] * net_dict['N_I']),
            }

# Total indegree of excitatory population
indegrees['CE'] = indegrees['CEE'] + indegrees['CEI']
# Total indegree of inhibitory population
indegrees['CI'] = indegrees['CII'] + indegrees['CIE']
# Total indegree of the model
indegrees['C_tot'] = indegrees['CE'] + indegrees['CI']

net_dict.update(indegrees)

# Create array of synaptic time constants for each neuron,
# ranging from 0.1 to 1.09 ms.
net_dict['tau_syn'] = [0.1 + 0.01 * i for i in range(net_dict['nr_ports'])]

net_dict['neuron_params'] = {
                            # Membrane capacitance (in pF)
                            "C_m": 1.0,
                            # Membrane time constant (in ms)
                            "tau_m": net_dict['tauMem'],
                            # Refractory period of neurons after spike (in ms)
                            "t_ref": 2.0,
                            # Reset membrane potential of the neurons (in mV)
                            "E_L": 0.0,
                            # Membrane potential after a spike (in mV)
                            "V_reset": 0.0,
                            "V_m": 0.0,
                            # Threshold potential of the neurons (in mV)
                            "V_th": net_dict['theta'],
                            # Time constant of postsynaptic currents (in ms)
                            "tau_syn": net_dict['tau_syn']
                            },

# Amplitude of excitatory postsynaptic current
net_dict['J_ex'] = net_dict['J']
# amplitude of inhibitory postsynaptic current
net_dict['J_in'] = -net_dict['g'] * net_dict['J']

# Threshold rate, which is the external rate needed to fix
# the membrane potential around its threshold
net_dict['nu_th'] = net_dict['theta'] / \
                    (net_dict['J'] * net_dict['CE'] * net_dict['tauMem'])

# External firing rate
net_dict['nu_ex'] = net_dict['eta'] * net_dict['nu_th']

# Rate of the poisson generator
# multiplied by the in-degree CE and converted to Hz by multiplication by 1000
net_dict['p_rate'] = 1000.0 * net_dict['nu_ex'] * net_dict['CE']

# Population connection parameters
net_dict['conn_params_ee'] = {'rule': 'fixed_indegree',
                              'indegree': net_dict['CEE']}
net_dict['conn_params_ie'] = {'rule': 'fixed_indegree',
                              'indegree': net_dict['CIE']}
net_dict['conn_params_ii'] = {'rule': 'fixed_indegree',
                              'indegree': net_dict['CII']}
net_dict['conn_params_ei'] = {'rule': 'fixed_indegree',
                              'indegree': net_dict['CEI']}
