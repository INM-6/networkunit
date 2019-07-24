"""Loads NetworkUnit model classes for NeuronUnit"""

import pkgutil

"""
NOTE: All test files must have a prefix "model_" and extension ".py".
Only these would be loaded.
"""

__path__ = pkgutil.extend_path(__path__, __name__)

for importer, modname, ispkg in pkgutil.walk_packages(path=__path__, prefix=__name__+'.'):
    base_modname = str.split(modname, '.')[-1]
    if '_' in base_modname:
        module_type, module_name = str.split(base_modname, '_', 1)
        if module_type == 'model':
            exec("from {} import {}".format(modname, module_name))
    elif base_modname == "backends":
        exec("import {}".format(modname))
