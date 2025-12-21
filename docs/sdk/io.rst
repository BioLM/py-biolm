``biolmai.io``
==============

The BioLM Python SDK provides utilities for converting between common biological file formats (FASTA, CSV, PDB) and BioLM API request/response JSON structures. This module enables easy loading of data from files and exporting API results.

Quick Start
-----------

**Loading sequences from FASTA:**

.. code-block:: python

    from biolmai.io import load_fasta
    from biolmai import Model
    
    # Load sequences from FASTA file
    items = load_fasta("sequences.fasta")
    
    # Use with Model
    model = Model("esm2-8m")
    results = model.encode(items=items)

**Exporting results to CSV:**

.. code-block:: python

    from biolmai.io import to_csv
    from biolmai import Model
    
    # Get predictions
    model = Model("esmfold")
    results = model.predict(items=[{"sequence": "ACDEFGHIKLMNPQRSTVWY"}])
    
    # Export to CSV
    to_csv(results, "output.csv")

**Loading PDB structures:**

.. code-block:: python

    from biolmai.io import load_pdb
    from biolmai import Model
    
    # Load PDB structure
    items = load_pdb("structure.pdb")
    
    # Use with structure prediction model
    model = Model("antifold")
    results = model.predict(items=items)

Usage
-----

.. include:: usage/io.rst

The io module supports the following formats:

- **FASTA**: Sequence files (DNA, RNA, protein)
- **CSV**: Tabular data with headers
- **PDB**: Protein structure files (single and multi-model)

Each format provides both ``load_*`` (file → API JSON) and ``to_*`` (API JSON → file) functions.

API Reference
-------------

.. automodule:: biolmai.io.fasta
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: biolmai.io.csv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: biolmai.io.pdb
   :members:
   :undoc-members:
   :show-inheritance:
