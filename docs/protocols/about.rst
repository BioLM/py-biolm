``about``
=========

The ``about`` field provides informational metadata for cataloging and citation of protocols. This field is **optional** but recommended for protocols that will be shared, published, or archived.

Overview
--------

The ``about`` field helps with:

- **Discovery**: Keywords and descriptions make protocols searchable
- **Attribution**: Author information ensures proper credit
- **Reproducibility**: DOIs and citations link to published work
- **Documentation**: Links provide access to related resources

When to Use
-----------

Include the ``about`` field when:
- Sharing protocols with others
- Publishing protocols in papers or repositories
- Archiving protocols for long-term use
- Building a protocol library or catalog

For internal or one-off protocols, you can omit this field entirely.

Schema Definition
-----------------

.. jsonschema:: ../../schema/protocol_schema.json
   :path: /properties/about

About Object Schema
-------------------

The ``about`` object contains structured metadata. All fields are optional, but including at least a ``title`` and ``description`` is recommended.

.. jsonschema:: ../../schema/protocol_schema.json
   :path: /$defs/About

Field Descriptions
------------------

**title** (string, optional)
  A concise title for the protocol. Should be descriptive but brief.

**description** (string, optional)
  A detailed description of what the protocol does, its purpose, and any important notes. This can be multiple paragraphs.

**authors** (array, optional)
  List of authors who created or contributed to the protocol. Each author can include:
  
  - ``name``: Full name
  - ``affiliation``: Institution or organization
  - ``email``: Contact email
  - ``orcid``: ORCID identifier (format: "0000-0000-0000-0000")

**keywords** (array, optional)
  Keywords for discovery and categorization. Use specific, relevant terms that describe the protocol's domain, methods, or applications.

**doi** (string, optional)
  Digital Object Identifier if the protocol is published. Format: "10.1234/example.doi"

**cite** (string, optional)
  BibTeX citation string for academic references. Include the full BibTeX entry.

**links** (object, optional)
  Named links to related resources. Common keys include:
  
  - ``github``: Link to source code or repository
  - ``publication``: Link to published paper
  - ``docs``: Link to additional documentation
  - ``dataset``: Link to associated datasets
  
  You can use any key names that make sense for your use case.

Examples
--------

Minimal About Section
~~~~~~~~~~~~~~~~~~~~~

For simple protocols, just include the essentials:

.. code-block:: yaml

   about:
     title: "Antibody Design Protocol"
     description: "A workflow for designing antibody sequences using AntiFold and IgBert scoring"
     keywords: ["antibody", "design", "protein"]

Complete About Section
~~~~~~~~~~~~~~~~~~~~~~~

For published or shared protocols, include comprehensive metadata:

.. code-block:: yaml

   about:
     title: "Antibody Design Protocol with Multi-Model Scoring"
     description: |
       A comprehensive workflow for antibody design that:
       
       - Generates candidate sequences using AntiFold
       - Scores sequences using IgBert
       - Filters and ranks results
       - Logs outputs to MLflow for tracking
       
       This protocol is optimized for high-throughput screening
       of antibody variants.
     authors:
       - name: "Jane Smith"
         affiliation: "BioLM Research"
         email: "jane@biolm.ai"
         orcid: "0000-0001-2345-6789"
       - name: "John Doe"
         affiliation: "BioLM Research"
         email: "john@biolm.ai"
         orcid: "0000-0002-3456-7890"
     keywords: ["antibody", "design", "protein", "therapeutics", "high-throughput"]
     doi: "10.1234/example.doi"
     cite: |
       @article{smith2024antibody,
         title={High-Throughput Antibody Design Protocol},
         author={Smith, Jane and Doe, John},
         journal={BioLM Protocols},
         year={2024},
         doi={10.1234/example.doi}
       }
     links:
       github: "https://github.com/biolm/protocols/antibody-design"
       publication: "https://example.com/papers/antibody-design"
       docs: "https://docs.biolm.ai/protocols/antibody-design"
