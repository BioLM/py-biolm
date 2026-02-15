``biolmai.models``
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

Usage
-----

.. include:: usage/models.rst

For more detailed information, see:

- :doc:`../getting-started/concepts` - Core concepts: BioLM vs BioLMApi, sync/async, batching, error handling, disk output, rate limiting

Generating Usage Examples
-------------------------

You can automatically generate SDK usage examples for any model using the ``get_example()`` function or the ``Model.get_example()`` method:

.. code-block:: python

    from biolmai import get_example, Model
    
    # Generate example for a specific model and action
    example = get_example("esm2-8m", action="encode", format="python")
    print(example)
    
    # Or use the Model class
    model = Model("esm2-8m")
    example = model.get_example(action="encode", format="python")
    
    # Generate examples for all supported actions
    all_examples = model.get_examples(format="python")

Supported output formats include ``python``, ``markdown``, ``rst``, and ``json``.

You can also list all available models:

.. code-block:: python

    from biolmai import list_models
    
    models = list_models()
    for model in models:
        print(f"{model.get('model_name')} ({model.get('model_slug')})")

API Reference
-------------

.. automodule:: biolmai.models
   :members:
   :show-inheritance:
   :undoc-members:
