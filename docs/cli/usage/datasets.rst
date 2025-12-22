Datasets
========

Working with datasets in the CLI.

The dataset commands provide a thin wrapper around MLflow artifact operations for managing datasets. Each dataset is represented as an MLflow run within a dedicated experiment (default: ``datasets``).

Overview
--------

Datasets are managed using MLflow runs with specific tags:
- Each dataset has a unique ``dataset_id``
- Datasets are stored in MLflow experiments (default: ``datasets``)
- Files are uploaded as MLflow artifacts
- Authentication is handled automatically via OAuth credentials

Installation
------------

MLflow support is an optional dependency. Install it with:

.. code-block:: bash

   pip install biolmai[mlflow]

If MLflow is not installed, dataset commands will show a helpful error message with installation instructions.

Authentication
--------------

Dataset commands require authentication. Make sure you're logged in:

.. code-block:: bash

   biolm login

The commands use your OAuth credentials from ``~/.biolmai/credentials`` automatically.

Listing Datasets
----------------

List all available datasets:

.. code-block:: bash

   biolm dataset list

List datasets in a specific experiment:

.. code-block:: bash

   biolm dataset list --experiment my-datasets

Output formats:

.. code-block:: bash

   # Table format (default)
   biolm dataset list

   # JSON format
   biolm dataset list --format json

   # CSV format
   biolm dataset list --format csv --output datasets.csv

Showing Dataset Details
-----------------------

Show detailed information about a specific dataset:

.. code-block:: bash

   biolm dataset show my-dataset-123

The output includes:
- Dataset metadata (ID, name, status, timestamps)
- Tags
- Parameters
- Metrics
- Artifacts list

Output formats:

.. code-block:: bash

   # Table format (default)
   biolm dataset show my-dataset-123

   # JSON format
   biolm dataset show my-dataset-123 --format json

   # YAML format
   biolm dataset show my-dataset-123 --format yaml --output dataset.yaml

Uploading Datasets
------------------

Upload a single file to a dataset:

.. code-block:: bash

   biolm dataset upload my-dataset-123 data.csv

If the dataset doesn't exist, it will be created automatically.

Upload a directory:

.. code-block:: bash

   biolm dataset upload my-dataset-123 ./data --recursive

Upload with a custom name:

.. code-block:: bash

   biolm dataset upload my-dataset-123 data.csv --name "Training Data v1"

The upload command supports:
- Single files (automatically detected)
- Directories (use ``--recursive`` flag)
- Custom dataset names (via ``--name`` option)
- Custom MLflow experiments (via ``--experiment`` option)

Downloading Datasets
--------------------

Download all artifacts from a dataset:

.. code-block:: bash

   biolm dataset download my-dataset-123

Download to a specific directory:

.. code-block:: bash

   biolm dataset download my-dataset-123 ./downloads

Download a specific artifact:

.. code-block:: bash

   biolm dataset download my-dataset-123 ./downloads --artifact-path model.pkl

The download command:
- Creates the output directory if it doesn't exist
- Preserves directory structure from MLflow
- Shows progress during download
- Displays the download location on completion

Common Workflows
----------------

Creating and Uploading a Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Upload your first file to create a new dataset:

.. code-block:: bash

   biolm dataset upload my-new-dataset data.csv

2. Add more files to the same dataset:

.. code-block:: bash

   biolm dataset upload my-new-dataset additional-data.csv

3. Verify the upload:

.. code-block:: bash

   biolm dataset show my-new-dataset

Working with Different File Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upload various file types:

.. code-block:: bash

   # CSV files
   biolm dataset upload dataset-1 data.csv

   # JSON files
   biolm dataset upload dataset-1 data.json

   # FASTA files
   biolm dataset upload dataset-1 sequences.fasta

   # Directories with multiple files
   biolm dataset upload dataset-1 ./data --recursive

Sharing and Downloading Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. List available datasets:

.. code-block:: bash

   biolm dataset list

2. Show details to verify contents:

.. code-block:: bash

   biolm dataset show dataset-123

3. Download the dataset:

.. code-block:: bash

   biolm dataset download dataset-123 ./my-downloads

Using Custom MLflow Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, datasets are stored in the ``datasets`` experiment. You can use custom experiments:

.. code-block:: bash

   # Upload to custom experiment
   biolm dataset upload my-dataset data.csv --experiment my-experiments

   # List datasets in custom experiment
   biolm dataset list --experiment my-experiments

   # Show dataset from custom experiment
   biolm dataset show my-dataset --experiment my-experiments

Troubleshooting
---------------

Dataset Not Found
~~~~~~~~~~~~~~~~~

If you get a "Dataset not found" error:

1. Verify the dataset ID is correct:

.. code-block:: bash

   biolm dataset list

2. Check if you're using the correct experiment:

.. code-block:: bash

   biolm dataset list --experiment datasets

Authentication Errors
~~~~~~~~~~~~~~~~~~~~~~

If you get authentication errors:

1. Check your login status:

.. code-block:: bash

   biolm status

2. Re-authenticate if needed:

.. code-block:: bash

   biolm login

MLflow Not Available
~~~~~~~~~~~~~~~~~~~~

If you get "MLflow Not Available" error:

1. Install MLflow support:

.. code-block:: bash

   pip install biolmai[mlflow]

2. Verify installation:

.. code-block:: bash

   python -c "import mlflow; print(mlflow.__version__)"

Upload/Download Errors
~~~~~~~~~~~~~~~~~~~~~~

If upload or download fails:

1. Check file permissions
2. Verify disk space
3. Check network connectivity
4. Verify MLflow server is accessible (check ``--mlflow-uri``)

For command reference, see :doc:`../reference` (auto-generated from CLI code).

