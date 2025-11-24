Authenticating
==============

Authentication with the CLI.

The CLI provides commands for managing your authentication credentials.

Login
-----

Authenticate with your BioLM account:

.. code-block:: bash

   biolmai login

You'll be prompted for your username and password. Upon successful authentication, your access and refresh tokens will be saved locally.

.. note::

   This method does not work with social logins (Google, GitHub). Only use if you registered with email and password.

Logout
------

Remove saved authentication credentials:

.. code-block:: bash

   biolmai logout

Status
------

Check your current authentication status and environment configuration:

.. code-block:: bash

   biolmai status

This displays:
- Your authentication token status
- Environment variables
- Configuration paths

For more details, see the :doc:`../reference` section.

