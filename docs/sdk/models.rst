``biolmai.Models``
==================

The BioLM Python SDK provides a high-level interface for working with BioLM models. This page covers usage examples, API reference, and best practices.

Quick Start
-----------

**Using the convenience function:**

.. code-block:: python

    from biolmai import biolm
    
    # Encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")
    
    # Predict a batch of sequences
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "SEQ2"])

**Using the Model class:**

.. code-block:: python

    from biolmai import Model
    
    model = Model("esm2-8m")
    result = model.encode(items=[{"sequence": "MSILVTRPSPAGEEL"}])
    
    model = Model("esmfold")
    result = model.predict(items=[{"sequence": "MDNELE"}])

Synchronous and Asynchronous Usage
-----------------------------------

.. include:: usage/async-sync.rst

Batching and Input Formats
---------------------------

.. include:: usage/batching.rst

Error Handling
--------------

.. include:: usage/error-handling.rst

API Reference
-------------

.. automodule:: biolmai.models
   :members:
   :show-inheritance:
   :undoc-members:
