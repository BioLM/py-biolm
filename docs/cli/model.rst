``biolm model``
===============

Work with BioLM models.

Usage
-----

The model commands allow you to explore available models, view model details, run models, and generate SDK usage examples.

Examples
--------

List all available models:

.. code-block:: bash

   biolm model list

Filter models by capabilities:

.. code-block:: bash

   # List only encoder models
   biolm model list --filter encoder=true
   
   # List models sorted by name
   biolm model list --sort model_name
   
   # Output as JSON
   biolm model list --format json --output models.json
   
   # Compact view with specific fields
   biolm model list --view compact --fields model_name,model_slug,actions

Show details for a specific model:

.. code-block:: bash

   # Show basic model information
   biolm model show esm2-8m
   
   # Include JSON schemas for each action
   biolm model show esmfold --include-schemas
   
   # Output as JSON
   biolm model show esm2-8m --format json --output model.json

Run a model:

.. code-block:: bash

   # Run model with FASTA input
   biolm model run esm2-8m encode -i sequences.fasta -o embeddings.json
   
   # Run with CSV input and parameters
   biolm model run esmfold predict -i data.csv --params '{"temperature": 0.7}'
   
   # Run with progress bar for large files
   biolm model run esm2-8m encode -i large.fasta --progress
   
   # Run with stdin input
   echo '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}' | biolm model run esm2-8m encode -i - --format json
   
   # Run with different output formats
   biolm model run esmfold predict -i sequences.fasta -o results.fasta --format fasta

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

Command Reference
-----------------

.. click:: biolmai.cli:model
   :prog: biolm model
   :show-nested:

