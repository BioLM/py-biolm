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

Run a model:

.. code-block:: bash

   biolm model run model-name --input data.txt

Command Reference
-----------------

.. click:: biolmai.cli:model
   :prog: biolm model
   :show-nested:

