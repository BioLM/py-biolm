Models
======

Working with models in the CLI.

Listing Available Models
-------------------------

The ``biolm model list`` command displays all available BioLM models with filtering, sorting, and various output format options.

Basic usage:

.. code-block:: bash

   biolm model list

Filter models by capabilities:

.. code-block:: bash

   # List only encoder models
   biolm model list --filter encoder=true
   
   # List only predictor models
   biolm model list --filter predictor=true
   
   # List models by name
   biolm model list --filter model_name=ESM

Sort models:

.. code-block:: bash

   # Sort by model name (ascending)
   biolm model list --sort model_name
   
   # Sort by model name (descending)
   biolm model list --sort -model_name

Output formats:

.. code-block:: bash

   # JSON output
   biolm model list --format json --output models.json
   
   # CSV output
   biolm model list --format csv --output models.csv
   
   # YAML output (requires PyYAML)
   biolm model list --format yaml --output models.yaml

Predefined views:

.. code-block:: bash

   # Compact view (name, slug, actions)
   biolm model list --view compact
   
   # Detailed view (includes capabilities)
   biolm model list --view detailed
   
   # Full view (all fields)
   biolm model list --view full

Viewing Model Details
---------------------

The ``biolm model show`` command displays detailed information about a specific model:

.. code-block:: bash

   # Show basic model information
   biolm model show esm2-8m
   
   # Include JSON schemas for each action
   biolm model show esmfold --include-schemas
   
   # Output as JSON
   biolm model show esm2-8m --format json --output model.json

The output includes:
- Model name and slug
- Available actions (encode, predict, generate, etc.)
- Model metadata and descriptions
- Optional JSON schemas for each action

Running Models
--------------

The ``biolm model run`` command executes a model with input data from files or stdin:

Input from files:

.. code-block:: bash

   # FASTA input
   biolm model run esm2-8m encode -i sequences.fasta -o embeddings.json
   
   # CSV input
   biolm model run esmfold predict -i data.csv -o predictions.json
   
   # JSON input
   biolm model run esm2-8m encode -i data.json -o results.json
   
   # PDB input
   biolm model run esmfold predict -i structure.pdb -o predictions.json

Input from stdin:

.. code-block:: bash

   # JSON from stdin
   echo '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}' | biolm model run esm2-8m encode -i - --format json
   
   # Multiple items from stdin
   cat data.jsonl | biolm model run esm2-8m encode -i - --format json

Output formats:

.. code-block:: bash

   # JSON output (default)
   biolm model run esm2-8m encode -i sequences.fasta -o results.json
   
   # FASTA output
   biolm model run esmfold predict -i sequences.fasta -o results.fasta --format fasta
   
   # CSV output
   biolm model run esm2-8m encode -i sequences.fasta -o results.csv --format csv

With parameters:

.. code-block:: bash

   # Parameters as JSON string
   biolm model run esm2-8m encode -i sequences.fasta --params '{"normalize": true}'
   
   # Parameters from file
   biolm model run esmfold predict -i sequences.fasta --params params.json

Batch processing:

.. code-block:: bash

   # Show progress bar for large files
   biolm model run esm2-8m encode -i large.fasta --progress
   
   # Specify batch size
   biolm model run esm2-8m encode -i large.fasta --batch-size 50

Supported file formats:

- **FASTA** (``.fasta``, ``.fa``, ``.fas``): Sequence files
- **CSV** (``.csv``): Comma-separated values
- **JSON** (``.json``): JSON arrays or objects
- **JSONL** (``.jsonl``): Newline-delimited JSON
- **PDB** (``.pdb``): Protein structure files

Format auto-detection:

The CLI automatically detects input and output formats from file extensions. You can override with the ``--format`` option.

Generating SDK Usage Examples
------------------------------

The ``biolm model example`` command generates ready-to-use SDK code examples for any model:

.. code-block:: bash

   # Generate a Python example for a model
   biolm model example esm2-8m
   
   # Generate example for a specific action
   biolm model example esmfold --action predict
   
   # Generate in different formats
   biolm model example ablang2 --format markdown
   biolm model example ablang2 --format rst
   biolm model example ablang2 --format json
   
   # Save example to a file
   biolm model example esm2-8m --output my_example.py

The generated examples show how to use the model with the SDK, including:

- Importing the necessary modules
- Creating a Model instance
- Calling the appropriate action method
- Using the correct input format

Examples are generated based on the actual model schemas, ensuring accuracy and correctness.

Common Workflows
----------------

Encode sequences from FASTA file:

.. code-block:: bash

   biolm model run esm2-8m encode -i sequences.fasta -o embeddings.json

Predict structures for multiple sequences:

.. code-block:: bash

   biolm model run esmfold predict -i sequences.fasta -o structures.json --progress

Filter and explore models:

.. code-block:: bash

   # Find all encoder models
   biolm model list --filter encoder=true
   
   # Show details for a specific model
   biolm model show esm2-8m --include-schemas
   
   # Generate SDK example
   biolm model example esm2-8m --action encode

Convert between formats:

.. code-block:: bash

   # FASTA to JSON
   biolm model run esm2-8m encode -i sequences.fasta -o results.json
   
   # CSV to FASTA
   biolm model run esmfold predict -i data.csv -o results.fasta --format fasta

Troubleshooting
---------------

**Model not found:**
   Use ``biolm model list`` to see all available models and their exact names/slugs.

**Format detection errors:**
   Specify the format explicitly with ``--format`` option.

**Empty input:**
   Ensure your input file contains valid data in the expected format.

**Authentication errors:**
   Run ``biolm login`` to authenticate, or check your credentials with ``biolm status``.

**Network errors:**
   Check your internet connection and API endpoint accessibility.

Output Formats
--------------

- **table** (default): Formatted table output
- **json**: JSON format
- **yaml**: YAML format (requires PyYAML)
- **csv**: CSV format
- **fasta**: FASTA format (for sequence data)
- **pdb**: PDB format (for structure data)

.. note::

   **Documentation authors:** This page contains usage examples and explanations for using model commands in the CLI. For command reference, see :doc:`../reference` (auto-generated from CLI code).

