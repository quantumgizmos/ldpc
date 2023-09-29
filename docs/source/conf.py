import os
import sys
import ldpc2
sys.path.insert(0, ldpc2.__file__)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LDPC'
copyright = '2023, Joschka Roffe'
author = 'Joschka Roffe'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme',"myst_parser"]

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


html_static_path = ['_static']

