``biolm status``
================

Show authentication status and configuration.

Usage
-----

Check your current authentication status and environment configuration:

.. code-block:: bash

   biolm status

This command displays:
- Environment variables (BIOLMAI_TOKEN, if set)
- Credentials file path
- API URL
- Authentication token validation status

Examples
--------

Check authentication status:

.. code-block:: bash

   biolm status

The command will show a formatted table with your configuration and validate any existing credentials.

Command Reference
-----------------

.. click:: biolmai.cli:status
   :prog: biolm status
