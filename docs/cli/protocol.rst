``biolm protocol``
==================

Work with protocols.

Usage
-----

The protocol commands allow you to manage and execute BioLM protocols.

Examples
--------

List all protocols:

.. code-block:: bash

   biolm protocol list

Show details for a specific protocol:

.. code-block:: bash

   biolm protocol show protocol-id

Run a protocol from a YAML file:

.. code-block:: bash

   biolm protocol run protocol.yaml

Validate a protocol YAML file:

.. code-block:: bash

   biolm protocol validate protocol.yaml

Command Reference
-----------------

.. click:: biolmai.cli:protocol
   :prog: biolm protocol
   :show-nested:

