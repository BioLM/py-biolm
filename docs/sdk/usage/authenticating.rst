Authenticating
==============

Authentication with the SDK.

Environment Variable
--------------------

Set the ``BIOLMAI_TOKEN`` environment variable:

.. code-block:: bash

   export BIOLMAI_TOKEN=your_api_token_here

Or in Python:

.. code-block:: python

   import os
   os.environ['BIOLMAI_TOKEN'] = 'your_api_token_here'

API Key Parameter
-----------------

Pass the API key directly when creating a client:

.. code-block:: python

   from biolmai.client import BioLMApi

   client = BioLMApi("esm2-8m", api_key="your_api_token_here")

For more information, see :doc:`../../getting-started/authentication`.

