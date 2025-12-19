Models
======

Working with models in the CLI.

Listing Available Models
-------------------------

To see all available BioLM models:

.. code-block:: bash

   biolm model example

This will display a table showing model names, slugs, and available actions.

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

Output Formats
--------------

- **python** (default): Python code ready to copy and paste
- **markdown**: Markdown formatted with code blocks
- **rst**: reStructuredText format for documentation
- **json**: Structured JSON with model metadata and code

.. note::

   **Documentation authors:** This page should contain usage examples and explanations for using model commands in the CLI. Add examples, common workflows, and troubleshooting tips here.

   For command reference, see :doc:`../reference` (auto-generated from CLI code).

