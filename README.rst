========
BioLM AI
========


.. image:: https://img.shields.io/pypi/v/biolm.svg
        :target: https://pypi.python.org/pypi/biolm

.. image:: https://img.shields.io/pypi/v/biolm.svg
        :target: https://pypi.python.org/pypi/biolm

.. image:: https://api.travis-ci.com/BioLM/py-biolm.svg?branch=production
        :target: https://travis-ci.org/github/BioLM/py-biolm

.. image:: https://readthedocs.org/projects/biolm-ai/badge/?version=latest
        :target: https://biolm-ai.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python client and SDK for `BioLM <https://biolm.ai>`_

.. include:: docs/python-client/quickstart.rst
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

.. include:: docs/python-client/overview.rst
   :start-line: 1
   :end-line: 17

.. include:: docs/python-client/features.rst

* Free software: Apache Software License 2.0
* Documentation: https://biolm-ai.readthedocs.io.