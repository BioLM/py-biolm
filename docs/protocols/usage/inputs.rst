``inputs``
==========

The ``inputs`` field defines input parameters for the protocol with default values. These parameters can be referenced in template expressions throughout the protocol.

Overview
--------

Inputs are defined as a mapping (dictionary) where:
- **Keys** are input parameter names
- **Values** can be any YAML type (string, number, boolean, array, object)
- **Values** may contain template expressions: ``${{ ... }}``

Schema Definition
-----------------

.. jsonschema:: ../../../schema/protocol_schema.json#/properties/inputs

Examples
--------

Basic Inputs
~~~~~~~~~~~~

.. code-block:: yaml

   inputs:
     pdb_str: string
     heavy_chain: string
     n_samples: 20
     temperature: 1.0
     regions: ["CDR1", "CDR2", "CDR3"]
     dev: false

Inputs with Template Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input values can use template expressions for computed defaults:

.. code-block:: yaml

   inputs:
     batch_size: 4
     total_items: 100
     num_batches: ${{ total_items // batch_size }}

Usage in Protocol
------------------

Input values can be referenced in template expressions throughout the protocol:

- In task parameters: ``${{ n_samples }}``
- In request bodies: ``${{ pdb_str }}``
- In expressions: ``${{ n_samples // execution.concurrency.workflow }}``
