``biolm model``
===============

Work with BioLM models.

Usage
-----

The model commands allow you to explore available models, view model details, and run models.

Examples
--------

List all available models:

.. code-block:: bash

   biolm model list

Show details for a specific model:

.. code-block:: bash

   biolm model show model-name

Generate SDK usage examples for a model:

.. code-block:: bash

   # List all available models
   biolm model example
   
   # Generate example for a specific model
   biolm model example esm2-8m
   
   # Generate example for a specific action
   biolm model example esm2-8m --action encode
   
   # Generate in different formats
   biolm model example esm2-8m --format markdown
   biolm model example esm2-8m --format rst
   biolm model example esm2-8m --format json
   
   # Save example to a file
   biolm model example esm2-8m --output example.py

Run a model:

.. code-block:: bash

   biolm model run model-name --input data.txt

Command Reference
-----------------

.. click:: biolmai.cli:model
   :prog: biolm model
   :show-nested:

