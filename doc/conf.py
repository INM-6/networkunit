import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
sys.path.insert(0, '..')
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NetworkUnit'
copyright = '2018, NetworkUnit authors and contributors'
author = 'Robin Gutzen, Michael von Papen, Michael Denker, Aitor Morales-Gregorio'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme_options = {
    'font_family': 'Arial',
    'page_width': '1200px',  # default is 940
    'sidebar_width': '280px',  # default is 220
    'logo': 'images/undefined.png',
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# latex_elements = {
#     'papersize' : 'letterpaper' ,
#     'pointsize' : '10pt' ,
#     'preamble' : '' ,
#     'figure_align' : 'htbp'
# }
