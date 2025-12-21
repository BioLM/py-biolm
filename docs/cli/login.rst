``biolm login``
===============

Login to BioLM using OAuth 2.0 with PKCE.

Usage
-----

Authenticate with your BioLM account:

.. code-block:: bash

   biolm login

The command will:
- Check for existing valid credentials
- If credentials are missing or invalid, open a browser for OAuth authorization
- Save credentials to ``~/.biolmai/credentials``

Options
-------

You can specify a custom OAuth client ID:

.. code-block:: bash

   biolm login --client-id your-client-id

Or set the ``BIOLMAI_OAUTH_CLIENT_ID`` environment variable.

You can also specify custom OAuth scopes:

.. code-block:: bash

   biolm login --scope "read write admin"

Examples
--------

Login with default settings:

.. code-block:: bash

   biolm login

Login with custom client ID:

.. code-block:: bash

   biolm login --client-id abc123xyz

Login with custom scope:

.. code-block:: bash

   biolm login --scope "read write"

Command Reference
-----------------

.. click:: biolmai.cli:login
   :prog: biolm login
