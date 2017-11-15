"""Loads NetworkUnit capability classes for NeuronUnit"""

from os.path import dirname, basename, isfile
import glob
from base_capabilities import *

"""
NOTE: All capability files must have a prefix "cap_" and extension ".py".
Only these would be loaded.
"""
files = glob.glob(dirname(__file__)+"/cap_*.py")
modules = [ basename(f)[:-3] for f in files if isfile(f)]

for module in modules:
   exec("from {} import *".format(module))
