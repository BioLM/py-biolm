``inputs``
==========

The ``inputs`` field defines input parameters for the protocol with default values. These parameters can be referenced in template expressions throughout the protocol using the syntax ``${{ input_name }}``.

Overview
--------

Inputs serve as the **interface** to your protocol, similar to function parameters. They allow:

- **Default values**: Provide sensible defaults that can be overridden
- **Type hints**: Indicate expected data types (though not strictly enforced)
- **Documentation**: Make it clear what inputs the protocol expects
- **Reusability**: Enable the same protocol to work with different inputs

Inputs are defined as a mapping (dictionary) where:
- **Keys** are input parameter names (must be valid identifiers)
- **Values** can be any YAML type (string, number, boolean, array, object)
- **Values** may contain template expressions: ``${{ ... }}``

**Inputs Properties**:

- **Type**: Object (dictionary/mapping)
- **Keys**: Input parameter names (must be valid identifiers)
- **Values**: Any YAML type (string, number, boolean, array, object) or template expressions
- **Evaluation**: Inputs are evaluated in order, so later inputs can reference earlier ones

How Inputs Work
---------------

Inputs are evaluated in two phases:

1. **Default assignment**: Input values are set from the ``inputs`` field
2. **Override**: Inputs can be overridden when the protocol is executed (via API or CLI)

Input values are then available throughout the protocol in template expressions.

Naming Conventions
------------------

Use descriptive, lowercase names with underscores:

- ✅ Good: ``n_samples``, ``temperature``, ``heavy_chain``
- ❌ Avoid: ``n``, ``temp``, ``h`` (too cryptic)

For boolean flags, use clear names:

- ✅ Good: ``dev``, ``enable_filtering``, ``skip_validation``
- ❌ Avoid: ``flag``, ``option``, ``mode`` (too vague)

Template Expressions in Inputs
-------------------------------

Input values can themselves use template expressions for computed defaults. This is useful for derived values:

.. code-block:: yaml

   inputs:
     batch_size: 4
     total_items: 100
     num_batches: ${{ total_items // batch_size }}  # Computed: 25

**Important**: When using expressions in inputs, the referenced inputs must be defined earlier in the same ``inputs`` object. The order matters for computed defaults.

Examples
--------

Basic Inputs
~~~~~~~~~~~~

Simple inputs with literal values:

.. code-block:: yaml

   inputs:
     pdb_str: string          # String placeholder
     heavy_chain: "A"          # Specific default value
     light_chain: "B"
     n_samples: 20            # Integer
     temperature: 1.0          # Float
     regions: ["CDR1", "CDR2", "CDR3"]  # Array
     dev: false                # Boolean

Inputs with Computed Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use template expressions to compute derived values:

.. code-block:: yaml

   inputs:
     n_samples: 20
     num_workers: 4
     samples_per_worker: ${{ n_samples // num_workers }}  # = 5
     max_batch_size: ${{ samples_per_worker * 2 }}       # = 10

Inputs with Complex Types
~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs can be complex objects:

.. code-block:: yaml

   inputs:
     chain_config:
       heavy: "A"
       light: "B"
     scoring_params:
       temperature: 1.0
       top_k: 10
     regions: ["CDR1", "CDR2", "CDR3"]

Using Inputs in Protocol
------------------------

Input values are referenced using template expressions throughout the protocol:

**In task parameters:**

.. code-block:: yaml

   request_body:
     params:
       num_seq_per_target: ${{ n_samples }}

**In request bodies:**

.. code-block:: yaml

   request_body:
     items:
       - pdb: ${{ pdb_str }}
         chain: ${{ heavy_chain }}

**In expressions:**

.. code-block:: yaml

   execution:
     concurrency:
       workflow: ${{ n_samples // 10 }}

**In conditional logic:**

.. code-block:: yaml

   - id: debug_task
     skip_if: "${{ not dev }}"

Best Practices
--------------

1. **Provide meaningful defaults**: Defaults should work for common use cases
2. **Use descriptive names**: Make it clear what each input represents
3. **Document expected types**: Use clear value types (e.g., ``string`` vs ``"example"``)
4. **Group related inputs**: Use objects for related parameters
5. **Avoid deep nesting**: Keep input structures relatively flat for clarity

Common Patterns
---------------

**Placeholder strings**: Use the literal string ``"string"`` to indicate a required string input:

.. code-block:: yaml

   inputs:
     pdb_str: string  # User must provide this

**Configuration objects**: Group related settings:

.. code-block:: yaml

   inputs:
     model_config:
       temperature: 1.0
       top_k: 10
       top_p: 0.9

**Lists of options**: Provide arrays as defaults:

.. code-block:: yaml

   inputs:
