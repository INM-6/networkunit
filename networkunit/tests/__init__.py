"""Loads NetworkUnit test classes"""

import pkgutil

"""
NOTE: All test files must have a prefix "test_" and extension ".py".
Only these would be loaded.
"""

__path__ = pkgutil.extend_path(__path__, __name__)

for importer, modname, ispkg in pkgutil.walk_packages(path=__path__, prefix=__name__+'.'):
    module_type, module_name = str.split(str.split(modname, '.')[-1], '_', 1)
    if module_type == 'test':
        exec("from {} import {}".format(modname, module_name))
