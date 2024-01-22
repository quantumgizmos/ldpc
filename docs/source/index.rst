.. LDPC documentation master file, created by
   sphinx-quickstart on Fri Sep 29 23:33:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LDPC's documentation!
================================

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Installation:

   installation.md

.. toctree::
   :maxdepth: 2
   :caption: Classical Coding:

   classical_coding.ipynb
   bp_decoding_example.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Quantum Coding:

   quantum_decoder.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Sinter Integration:

   sinter_integration.ipynb

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: ldpc API:

   ldpc/codes
   ldpc/mod2
   ldpc/code_util
   ldpc/bp_decoder
   ldpc/belief_find_decoder
   ldpc/bposd_decoder
   ldpc/union_find_decoder
   ldpc/monte_carlo_simulation
   ldpc/sinter_decoders.rst
   ldpc/soft_info_belief_propagation.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
