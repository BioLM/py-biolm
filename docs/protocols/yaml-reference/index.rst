YAML Reference
==============

Complete reference documentation for Protocol YAML schema. This page is **auto-generated** from the JSON Schema definition.

.. note::

   This documentation is automatically generated from the Protocol JSON Schema.
   To update it, modify ``schema/protocol_schema.json`` and rebuild the docs.

Schema Overview
---------------

Protocols are validated against a formal JSON Schema (Draft 2020-12) that defines:

- **Top-level structure**: Required and optional fields
- **Field types**: Data types and constraints for each field
- **Validation rules**: Conditional requirements, enum values, patterns
- **Task definitions**: Model tasks and gather tasks
- **Expression helpers**: Reusable template expression types

Top-Level Schema
----------------

**Required Fields:**
  - ``name`` (string) - Protocol name identifier
  - ``schema_version`` (integer) - Schema version (defaults to 1)
  - ``inputs`` (object) - Input parameter definitions with default values
  - ``tasks`` (array) - List of workflow tasks to execute

**Optional Fields:**
  - ``description`` (string) - Human-readable description
  - ``about`` (object) - Informational fields for cataloging and citation
  - ``protocol_version`` (string) - Protocol version identifier
  - ``outputs`` (array) - Output rules for MLflow logging
  - ``execution`` (object) - Execution configuration

.. toctree::
   :maxdepth: 2
   :caption: Schema Sections:

   top-level-fields
   tasks
   outputs
   execution

Template Expressions
--------------------

Template expressions use the syntax: ``${{ expression }}``

**Examples:**
  - ``${{ n_samples }}`` - Variable reference
  - ``${{ n_samples // execution.concurrency.workflow }}`` - Expression with operators
  - ``${{ sequences_batches_scoring.results }}`` - Nested property access
  - ``${{ item.designed_pdb }}`` - Item context in foreach loops
  - ``"${{ response.results[*].log_prob }}"`` - JSONPath expression

**Validation Rules:**
  1. **Syntax Validation**: Must match pattern ``^\$\{\{.*\}\}$``
  2. **Expression Validation**: 
     - Basic syntax checking (balanced braces, valid operators)
     - Variable references should reference valid ``inputs`` or task IDs
     - JSONPath expressions in ``response_mapping`` should be valid JSONPath syntax
  3. **Type Checking**: 
     - Cannot validate types at schema validation time (requires runtime)
     - Schema validation should check syntax only

**JSONPath in Response Mapping:**

Response mapping values use JSONPath-like syntax to extract data from API responses:
  - ``"${{ response.results[*].field }}"`` - Extract field from all results
  - ``"${{ response.results[*].sequences[*].heavy }}"`` - Nested array access
  - ``"${{ response.results[0].field }}"`` - Single result access

Validation Strategy
-------------------

**Phase 1: Structural Validation (JSON Schema)**
  - Validate YAML syntax
  - Validate required fields
  - Validate data types
  - Validate enum values
  - Validate conditional requirements (oneOf patterns)

**Phase 2: Semantic Validation (Custom Python)**
  - Validate template expression syntax
  - Validate task ID references in ``depends_on``, ``from``, ``foreach``, ``outputs``
  - Validate JSONPath syntax in ``response_mapping``
  - Validate slug/action mappings (requires API lookup)
  - Validate class/app/method combinations (requires API lookup)
  - Check for circular dependencies in task graph
  - Validate that ``outputs`` references valid task IDs and fields

**Phase 3: Runtime Validation (During Execution)**
  - Validate template expression evaluation
  - Validate API request/response schemas
  - Validate actual data types at runtime

For more information about using protocols, see the :doc:`../usage/inputs`, :doc:`../usage/tasks`, :doc:`../usage/outputs`, and :doc:`../usage/settings` sections.
