.. _quickstart-sdk:

==========
Quickstart
==========

Install the package:

.. code-block:: bash

    pip install biolm

Basic usage:

.. code-block:: python

    from biolm import BioLM

    # Encode a single sequence
    result = BioLM(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # Predict a batch of sequences
    result = BioLM(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

    # Write results to disk
    BioLM(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

For advanced usage, see :doc:`usage`.