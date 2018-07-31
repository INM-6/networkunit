"""Loads NetworkUnit plot classes for NeuronUnit"""

import pkgutil
import matplotlib.colors as colors

def alpha(color_inst, a):
    if color_inst[0] == '#':
        color_inst = colors.hex2color(color_inst)
    return [el + (1. - el) * (1 - a) for el in color_inst]


"""
NOTE: All test files must have a prefix "plot_" and extension ".py".
Only these would be loaded.
"""

__path__ = pkgutil.extend_path(__path__, __name__)

for importer, modname, ispkg in pkgutil.walk_packages(path=__path__, prefix=__name__+'.'):
    module_type, module_name = str.split(str.split(modname, '.')[-1], '_', 1)
    if module_type == 'plot':
        exec("from {} import {}".format(modname, module_name))
