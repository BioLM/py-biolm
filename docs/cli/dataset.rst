``biolm dataset``
=================

Manage datasets.

Usage
-----

The dataset commands allow you to create, list, upload, download, and manage datasets using MLflow.

Examples
--------

List all datasets:

.. code-block:: bash

   biolm dataset list

Show details for a specific dataset:

.. code-block:: bash

   biolm dataset show dataset-id

Upload data to a dataset:

.. code-block:: bash

   biolm dataset upload dataset-id data.csv

Download a dataset:

.. code-block:: bash

   biolm dataset download dataset-id ./downloads

Command Reference
-----------------

.. click:: biolmai.cli:dataset
   :prog: biolm dataset
   :show-nested:

