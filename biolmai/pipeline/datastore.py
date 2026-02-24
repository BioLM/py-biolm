"""
Backward-compatibility shim.

Old code imported from ``biolmai.pipeline.datastore``; the implementation has
moved to ``biolmai.pipeline.datastore_duckdb``.  Import ``DataStore`` from
either location — they resolve to the same class.
"""

from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore  # noqa: F401

__all__ = ["DataStore"]
