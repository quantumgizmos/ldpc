# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
import ldpc
import shutil

try: shutil.rmtree('../build')
except FileNotFoundError: pass
sys.path.insert(0, ldpc.get_include())

# -- Project information -----------------------------------------------------

project = 'LDPC'
copyright = '2022, Joschka Roffe'
author = 'Joschka Roffe'

# The full version, including alpha/beta/rc tags
release = ldpc.__version__
version=release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon',"myst_parser",'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'display_version': True,
    'style_nav_header_background': '#800020',
      'includehidden': True,
}

html_context = {
  'display_github': True,
  'github_repo': 'ldpc',
  'github_user': 'quantumgizmos',
  'github_url': 'https://github.com/quantumgizmos/ldpc'
}

rst_prolog = """
:github_url: https://github.com/quantumgizmos/ldpc
"""

nbsphinx_prolog = """
:github_url: https://github.com/quantumgizmos/ldpc
"""


# Sidebars
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']