``biolm dataset``
=================

Manage datasets.

Usage
-----

The dataset commands allow you to create, list, and manage datasets.

Examples
--------

List all datasets:

.. code-block:: bash

   biolm dataset list

Show details for a specific dataset:

.. code-block:: bash

   biolm dataset show dataset-id

Create a new dataset:

.. code-block:: bash

   biolm dataset create my-dataset

Upload data to a dataset:

.. code-block:: bash

   biolm dataset upload dataset-id data.csv

Command Reference
-----------------

.. click:: biolmai.cli:dataset
   :prog: biolm dataset
   :show-nested:

