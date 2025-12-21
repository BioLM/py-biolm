``biolm workspace``
===================

Manage workspaces.

Usage
-----

The workspace commands allow you to create, list, and manage BioLM workspaces.

Examples
--------

List all workspaces:

.. code-block:: bash

   biolm workspace list

Show details for a specific workspace:

.. code-block:: bash

   biolm workspace show workspace-id

Create a new workspace:

.. code-block:: bash

   biolm workspace create my-workspace

Delete a workspace:

.. code-block:: bash

   biolm workspace delete workspace-id

Command Reference
-----------------

.. click:: biolmai.cli:workspace
   :prog: biolm workspace
   :show-nested:

