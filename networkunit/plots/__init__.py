"""Loads NetworkUnit plot classes"""

from .correlation_matrix import correlation_matrix
from .covar_pdf_ei import covar_pdf_ei
from .covar_pdf import covar_pdf
from .eigenvalues import eigenvalues
from .eigenvector_loads import eigenvector_loads
from .mu_std_table import mu_std_table
from .power_spectral_density import power_spectral_density
from .rasterplot import rasterplot
from .sample_histogram import sample_histogram


import matplotlib.colors as colors

def alpha(color_inst, a):
    if color_inst[0] == '#':
        color_inst = colors.hex2color(color_inst)
    return [el + (1. - el) * (1 - a) for el in color_inst]
