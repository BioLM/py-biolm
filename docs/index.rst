===================================
Welcome to Our Python Client Docs
===================================

.. include:: python-client/quickstart.rst
   :start-line: 1
   :end-line: 27

Asynchronous usage:

.. code-block:: python

    from biolm.client import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())

.. include:: python-client/overview.rst
   :start-line: 1
   :end-line: 17

.. toctree::
   :maxdepth: 2
   :caption: Python Client:
   :hidden:

   python-client/installation
   python-client/authentication
   python-client/quickstart
   python-client/overview
   python-client/features
   python-client/usage
   python-client/batching
   python-client/error_handling
   python-client/rate_limiting
   python-client/disk_output
   python-client/async_sync
   python-client/api_biolm
   python-client/api_client
   python-client/faq
   modules.rst

.. toctree::
   :maxdepth: 2
   :caption: REST API:
   :hidden:

   api_docs/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Guides:
   :hidden:

   tutorials_use_cases/notebooks

.. toctree::
   :maxdepth: 2
   :caption: Documentation:
   :hidden:

   model-docs/esm2/index
   model-docs/esm1v/index
   model-docs/esmfold/index
   model-docs/esmif/index
   model-docs/biolmtox/index
   model-docs/progen2/index
   model-docs/ablang/index
   model-docs/proteinfer/index
   model-docs/protgpt2/index
   model-docs/dnabert/index
   model-docs/finetuning/index

