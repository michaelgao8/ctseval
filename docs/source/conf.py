import os
import sys
sys.path.insert(0, os.path.abspath('../../ctseval'))

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'ctseval'
copyright = '2024, Michael Gao'
author = 'Michael Gao'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}