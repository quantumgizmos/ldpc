import os
import sys
import ldpc2
import udlr
ldpc2_path = os.path.dirname(ldpc2.__file__)
udlr_path = os.path.dirname(udlr.__file__)

sys.path.insert(0, [ldpc2_path,udlr_path] )

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LDPC2'
copyright = '2023, Joschka Roffe'
author = 'Joschka Roffe'
version = ldpc2.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme','myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'display_version': True,
    'style_nav_header_background': '#00147e',
    'includehidden': True,
}

html_context = {
  'display_github': True,
  'github_repo': 'ldpc',
  'github_user': 'quantumgizmos',
  'github_url': 'https://github.com/quantumgizmos/ldpc/docs'
}

# Sidebars
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

html_static_path = ['_static']

