``biolm logout``
================

Logout and remove saved credentials.

Usage
-----

Remove saved authentication credentials:

.. code-block:: bash

   biolm logout

This command removes the saved credentials file at ``~/.biolmai/credentials``.
After logout, you will need to run ``biolm login`` again to authenticate.

Examples
--------

Logout from BioLM:

.. code-block:: bash

   biolm logout

If you're already logged out, the command will complete silently.

Command Reference
-----------------

.. click:: biolmai.cli:logout
   :prog: biolm logout
