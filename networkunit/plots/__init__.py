"""Loads NetworkUnit plot classes for NeuronUnit"""

from os.path import dirname, basename, isfile
import glob
import matplotlib.colors as colors

def alpha(color_inst, a):
    if color_inst[0] == '#':
        color_inst = colors.hex2color(color_inst)
    return [el + (1. - el) * (1 - a) for el in color_inst]

"""
NOTE: All plot files must have a prefix "plot_" and extension ".py".
Only these would be loaded.
"""
files = glob.glob(dirname(__file__)+"/plot_*.py")
modules = [ basename(f)[:-3] for f in files if isfile(f)]

for module in modules:
    exec("from %s import *" % module)