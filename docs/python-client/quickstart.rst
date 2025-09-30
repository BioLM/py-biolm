.. _quickstart-sdk:

==========
Quickstart
==========

Install the package:

.. code-block:: bash

    pip install biolmai

Basic usage:

.. code-block:: python

    from biolmai import biolm

    # Encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Predict a batch of sequences
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

    # Write results to disk
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

For advanced usage, see :doc:`usage`.