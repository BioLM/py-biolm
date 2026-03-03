"""
DuckDB-based DataStore with Parquet backend for efficient large-scale data management.

Key features:
- Columnar storage (Parquet)
- Out-of-core queries (bigger than RAM)
- Vectorized anti-join deduplication
- 5-50× faster than pandas for aggregations
- Diff-mode friendly batch inserts
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import duckdb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from biolmai.pipeline.base import WorkingSet


class DuckDBDataStore:
    """
    DuckDB + Parquet based datastore for efficient sequence management.

    Optimized for:
    - Millions of sequences
    - Complex queries (joins, filters, aggregations)
    - Out-of-core operations (bigger than RAM)
    - Diff mode with efficient deduplication

    Args:
        db_path: Path to DuckDB database file
        data_dir: Directory for Parquet files and large data

    Example:
        >>> ds = DuckDBDataStore("pipeline.db", "data/")
        >>> seq_id = ds.add_sequence("MKLLIV")
        >>> ds.add_prediction(seq_id, "tm", "temberture-regression", 65.5)
        >>>
        >>> # Efficient query - no memory explosion
        >>> high_tm = ds.query("SELECT * FROM predictions WHERE value > 60")
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "pipeline.duckdb",
        data_dir: Union[str, Path] = "./pipeline_data",
    ):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create Parquet directories
        self.sequences_dir = self.data_dir / "sequences"
        self.predictions_dir = self.data_dir / "predictions"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.structures_dir = self.data_dir / "structures"

        for d in [
            self.sequences_dir,
            self.predictions_dir,
            self.embeddings_dir,
            self.structures_dir,
        ]:
            d.mkdir(exist_ok=True)

        # Connect to DuckDB (persistent)
        self.conn = duckdb.connect(str(self.db_path))

        # Initialize schema
        self._init_schema()

    def _init_schema(self):
        """Initialize DuckDB tables backed by Parquet files."""

        # Sequences table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id INTEGER PRIMARY KEY,
                sequence VARCHAR NOT NULL,
                length INTEGER,
                hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create unique index for deduplication
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sequence_hash
            ON sequences(hash)
        """
        )

        # Predictions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                prediction_type VARCHAR,
                value DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indexes for fast queries
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_seq
            ON predictions(sequence_id)
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_type
            ON predictions(prediction_type, model_name)
        """
        )

        # Embeddings table — arrays stored inline as FLOAT[] (no per-file Parquet overhead)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                layer INTEGER,
                values FLOAT[],
                dimension INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        # Migration: add values column to existing databases that have embedding_path only
        try:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN values FLOAT[]")
        except Exception:
            pass  # Column already exists

        # Structures table — stores inline structure content (PDB/CIF strings)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                format VARCHAR,
                structure_path VARCHAR,
                structure_str TEXT,
                plddt DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        # Migration: add structure_str column to existing databases
        try:
            self.conn.execute("ALTER TABLE structures ADD COLUMN structure_str TEXT")
        except Exception:
            pass  # Column already exists
        # Migration: add compressed structure_data BLOB column (replaces uncompressed TEXT)
        try:
            self.conn.execute("ALTER TABLE structures ADD COLUMN structure_data BLOB")
        except Exception:
            pass  # Column already exists

        # Pipeline runs table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id VARCHAR PRIMARY KEY,
                pipeline_type VARCHAR,
                config VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """
        )

        # Stage completions table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_completions (
                stage_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                stage_name VARCHAR,
                status VARCHAR,
                input_count INTEGER,
                output_count INTEGER,
                completed_at TIMESTAMP
            )
        """
        )

        # Pipeline metadata (clustering results, stage diagnostics, etc.)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Generation metadata (parameters used to produce each generated sequence)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_metadata (
                metadata_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                temperature DOUBLE,
                top_k INTEGER,
                top_p DOUBLE,
                num_return_sequences INTEGER,
                do_sample BOOLEAN,
                repetition_penalty DOUBLE,
                max_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_gen_meta_seq
            ON generation_metadata(sequence_id)
        """
        )

        # Initialize sequence counter
        result = self.conn.execute("SELECT MAX(sequence_id) FROM sequences").fetchone()
        self._sequence_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(prediction_id) FROM predictions"
        ).fetchone()
        self._prediction_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(embedding_id) FROM embeddings"
        ).fetchone()
        self._embedding_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(structure_id) FROM structures"
        ).fetchone()
        self._structure_counter = (result[0] or 0) + 1

        result = self.conn.execute(
            "SELECT MAX(metadata_id) FROM generation_metadata"
        ).fetchone()
        self._generation_metadata_counter = (result[0] or 0) + 1

        # Filter results table — records which sequence_ids passed each filter stage
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filter_results (
                filter_id    INTEGER PRIMARY KEY,
                run_id       TEXT NOT NULL,
                stage_name   TEXT NOT NULL,
                sequence_id  INTEGER NOT NULL,
                passed       BOOLEAN NOT NULL DEFAULT TRUE,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_filter_results_lookup
            ON filter_results(run_id, stage_name)
        """
        )
        result = self.conn.execute(
            "SELECT MAX(filter_id) FROM filter_results"
        ).fetchone()
        self._filter_id_counter = (result[0] or 0) + 1

        # Sequence attributes table — stores extra per-sequence columns from
        # the input DataFrame (e.g. heavy_chain, light_chain for antibody models).
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sequence_attributes (
                sequence_id  INTEGER NOT NULL,
                attr_name    TEXT NOT NULL,
                attr_value   TEXT,
                PRIMARY KEY (sequence_id, attr_name)
            )
        """
        )

        # Pipeline context table — inter-stage shared key-value store
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_context (
                run_id  TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, key)
            )
        """
        )

        # Track which extra columns have been added to the sequences table
        self._extra_columns: set[str] = set()
        # Discover columns already present (for resume / re-open)
        cols_info = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sequences'"
        ).fetchall()
        base_cols = {"sequence_id", "sequence", "length", "hash", "created_at"}
        for (col_name,) in cols_info:
            if col_name not in base_cols:
                self._extra_columns.add(col_name)

    @staticmethod
    def _hash_sequence(sequence: str) -> str:
        """Create hash of sequence for deduplication."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_row(values: dict[str, str]) -> str:
        """Hash multiple column values for multi-column deduplication.

        Columns are sorted alphabetically; values joined with ``\\x00``.
        """
        parts = [str(values.get(c, "")) for c in sorted(values.keys())]
        payload = "\x00".join(parts)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def ensure_input_columns(self, columns: list[str]):
        """Ensure the ``sequences`` table has all the given columns.

        Uses ``ALTER TABLE ADD COLUMN`` for any that don't already exist.
        This is idempotent — safe to call on every pipeline run.
        """
        for col in columns:
            if col in self._extra_columns:
                continue
            # Don't add columns that are part of the base schema
            if col in ("sequence_id", "sequence", "length", "hash", "created_at"):
                continue
            try:
                self.conn.execute(
                    f'ALTER TABLE sequences ADD COLUMN "{col}" TEXT'
                )
            except Exception:
                pass  # Column already exists
            self._extra_columns.add(col)

    def add_sequences_batch(
        self,
        sequences: Optional[list[str]] = None,
        deduplicate: bool = True,
        input_df: Optional[pd.DataFrame] = None,
        input_columns: Optional[list[str]] = None,
    ) -> list[int]:
        """
        Add multiple sequences efficiently using anti-join deduplication.

        This is the RECOMMENDED way to add sequences - vectorized and fast!

        Two calling conventions:

        1. **Legacy** (sequence-only):
           ``add_sequences_batch(["MKLLIV", ...])``

        2. **Multi-column** (arbitrary input columns):
           ``add_sequences_batch(input_df=df, input_columns=["heavy_chain", "light_chain"])``
           In this mode, the hash is computed across all input columns, and
           the column values are stored directly on the ``sequences`` table.
           A ``sequence`` column is still written (concatenation of all
           input columns joined with ``:``) so downstream code has a fallback.

        Args:
            sequences: List of sequence strings (legacy path).
            deduplicate: Use anti-join to skip existing sequences.
            input_df: DataFrame with input columns (multi-column path).
            input_columns: Column names in *input_df* to use as primary data.

        Returns:
            List of sequence_ids (new and existing), preserving input order.
        """
        # ----- Multi-column path ------------------------------------------------
        if input_df is not None and input_columns is not None:
            if input_df.empty:
                return []

            # Validate input columns exist in DataFrame
            missing = [c for c in input_columns if c not in input_df.columns]
            if missing:
                raise ValueError(
                    f"input_df missing columns: {missing}. "
                    f"Available: {list(input_df.columns)}"
                )

            # Ensure sequences table has the needed columns
            self.ensure_input_columns(input_columns)

            # Build df_new with hash across all input columns
            df_new = input_df[input_columns].copy()
            df_new["hash"] = df_new.apply(
                lambda row: self._hash_row(
                    {c: row[c] for c in input_columns}
                ),
                axis=1,
            )
            # Synthesize a 'sequence' column for compatibility
            if "sequence" in input_columns:
                df_new["sequence"] = input_df["sequence"]
            else:
                df_new["sequence"] = df_new[input_columns].astype(str).agg(
                    ":".join, axis=1
                )
            df_new["length"] = df_new["sequence"].str.len()

            # Dedup within batch
            original_hashes = df_new["hash"].tolist()
            df_new = df_new.drop_duplicates(subset=["hash"], keep="first")

            self.conn.register("_seq_batch", df_new)
            try:
                if deduplicate:
                    df_to_insert = self.conn.execute(
                        """
                        SELECT s.*
                        FROM _seq_batch s
                        LEFT JOIN sequences t ON s.hash = t.hash
                        WHERE t.hash IS NULL
                    """
                    ).df()
                else:
                    df_to_insert = df_new.copy()

                if len(df_to_insert) > 0:
                    start_id = self._sequence_counter
                    df_to_insert["sequence_id"] = range(
                        start_id, start_id + len(df_to_insert)
                    )
                    df_to_insert["created_at"] = datetime.now()

                    # Build column list for INSERT
                    base_cols = ["sequence_id", "sequence", "length", "hash", "created_at"]
                    extra = [c for c in input_columns if c not in base_cols and c != "sequence"]
                    all_cols = base_cols + extra
                    col_list = ", ".join(f'"{c}"' for c in all_cols)
                    sel_list = ", ".join(f'"{c}"' for c in all_cols)

                    self.conn.register("_seq_insert", df_to_insert)
                    try:
                        self.conn.execute(
                            f"INSERT INTO sequences ({col_list}) SELECT {sel_list} FROM _seq_insert"
                        )
                    finally:
                        self.conn.unregister("_seq_insert")

                    self._sequence_counter += len(df_to_insert)

                df_result = self.conn.execute(
                    """
                    SELECT hash, sequence_id
                    FROM sequences
                    WHERE hash IN (SELECT hash FROM _seq_batch)
                """
                ).df()
            finally:
                self.conn.unregister("_seq_batch")

            hash_to_id = dict(zip(df_result["hash"], df_result["sequence_id"]))
            return [hash_to_id[h] for h in original_hashes]

        # ----- Legacy single-column path ----------------------------------------
        if not sequences:
            return []

        # Create DataFrame for incoming batch
        df_new = pd.DataFrame(
            {
                "sequence": sequences,
                "length": [len(s) for s in sequences],
                "hash": [self._hash_sequence(s) for s in sequences],
            }
        )

        # Remove duplicates within the batch
        df_new = df_new.drop_duplicates(subset=["hash"], keep="first")

        # Register df_new explicitly — DuckDB scope introspection is fragile in async
        self.conn.register("_seq_batch", df_new)
        try:
            if deduplicate:
                # Anti-join: find sequences NOT already in database
                df_to_insert = self.conn.execute(
                    """
                    SELECT s.*
                    FROM _seq_batch s
                    LEFT JOIN sequences t ON s.hash = t.hash
                    WHERE t.hash IS NULL
                """
                ).df()
            else:
                df_to_insert = df_new.copy()

            # Assign IDs
            if len(df_to_insert) > 0:
                start_id = self._sequence_counter
                df_to_insert["sequence_id"] = range(
                    start_id, start_id + len(df_to_insert)
                )
                df_to_insert["created_at"] = datetime.now()

                # Batch insert (vectorized!) - specify column order explicitly
                self.conn.register("_seq_insert", df_to_insert)
                try:
                    self.conn.execute(
                        """
                        INSERT INTO sequences (sequence_id, sequence, length, hash, created_at)
                        SELECT sequence_id, sequence, length, hash, created_at FROM _seq_insert
                    """
                    )
                finally:
                    self.conn.unregister("_seq_insert")

                self._sequence_counter += len(df_to_insert)

            # Get all IDs (including existing)
            df_result = self.conn.execute(
                """
                SELECT hash, sequence_id
                FROM sequences
                WHERE hash IN (SELECT hash FROM _seq_batch)
            """
            ).df()
        finally:
            self.conn.unregister("_seq_batch")

        # Map back to original order
        hash_to_id = dict(zip(df_result["hash"], df_result["sequence_id"]))
        return [hash_to_id[self._hash_sequence(seq)] for seq in sequences]

    def add_sequence(self, sequence: str) -> int:
        """Add single sequence (convenience wrapper)."""
        return self.add_sequences_batch([sequence])[0]

    def add_predictions_batch(self, data: list[dict[str, Any]]):
        """
        Batch add predictions efficiently.

        Args:
            data: List of dicts with keys: sequence_id, prediction_type,
                  model_name, value, metadata (optional)

        Example:
            >>> ds.add_predictions_batch([
            ...     {'sequence_id': 1, 'prediction_type': 'tm',
            ...      'model_name': 'temberture-regression', 'value': 65.5},
            ...     {'sequence_id': 2, 'prediction_type': 'tm',
            ...      'model_name': 'temberture-regression', 'value': 70.2},
            ... ])
        """
        if not data:
            return

        df = pd.DataFrame(data)

        # Assign IDs
        start_id = self._prediction_counter
        df["prediction_id"] = range(start_id, start_id + len(df))
        df["created_at"] = datetime.now()

        # Ensure metadata is JSON string
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
        else:
            df["metadata"] = None

        # Batch insert — register explicitly to avoid fragile DuckDB scope introspection
        self.conn.register("_pred_batch", df)
        try:
            self.conn.execute(
                """
                INSERT INTO predictions (prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at)
                SELECT prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at FROM _pred_batch
            """
            )
            self._prediction_counter += len(df)
        finally:
            self.conn.unregister("_pred_batch")

    def add_prediction(
        self,
        sequence_id: int,
        prediction_type: str,
        model_name: str,
        value: Optional[float],
        metadata: Optional[dict] = None,
    ):
        """Add single prediction (convenience wrapper)."""
        self.add_predictions_batch(
            [
                {
                    "sequence_id": sequence_id,
                    "prediction_type": prediction_type,
                    "model_name": model_name,
                    "value": value,
                    "metadata": metadata,
                }
            ]
        )

    def has_prediction(
        self, sequence: str, prediction_type: str, model_name: str
    ) -> bool:
        """Check if prediction exists for sequence."""
        seq_hash = self._hash_sequence(sequence)
        result = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.sequence_id
            WHERE s.hash = ?
            AND p.prediction_type = ?
            AND p.model_name = ?
        """,
            [seq_hash, prediction_type, model_name],
        ).fetchone()

        return result is not None and result[0] > 0

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """
        Execute arbitrary SQL query and return DataFrame.

        This is the POWER feature - query directly without loading everything!

        Args:
            sql: DuckDB SQL query
            params: Optional query parameters

        Returns:
            DataFrame with results (only loads what matches query!)

        Example:
            >>> # Find high-quality long sequences
            >>> df = ds.query('''
            ...     SELECT s.sequence, p.value as plddt
            ...     FROM sequences s
            ...     JOIN predictions p ON s.sequence_id = p.sequence_id
            ...     WHERE s.length > 200
            ...     AND p.prediction_type = 'plddt'
            ...     AND p.value > 80
            ... ''')
        """
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def get_predictions_by_sequence(
        self,
        sequence: str,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get predictions for a sequence."""
        seq_hash = self._hash_sequence(sequence)

        sql = """
            SELECT p.*
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params = [seq_hash]

        if prediction_type:
            sql += " AND p.prediction_type = ?"
            params.append(prediction_type)

        if model_name:
            sql += " AND p.model_name = ?"
            params.append(model_name)

        return self.conn.execute(sql, params).df()

    def get_sequence_id(self, sequence: str) -> Optional[int]:
        """Get sequence_id for a sequence."""
        seq_hash = self._hash_sequence(sequence)
        result = self.conn.execute(
            "SELECT sequence_id FROM sequences WHERE hash = ?", [seq_hash]
        ).fetchone()
        return result[0] if result else None

    def add_embedding(
        self,
        sequence_id: int,
        model_name: str,
        embedding: np.ndarray,
        layer: Optional[int] = None,
    ):
        """
        Add embedding stored inline in DuckDB as FLOAT[] (no per-file Parquet overhead).

        Args:
            sequence_id: Sequence ID
            model_name: Model name
            embedding: Numpy array
            layer: Optional layer number
        """
        embedding_id = self._embedding_counter
        self._embedding_counter += 1

        self.conn.execute(
            """
            INSERT INTO embeddings
            (embedding_id, sequence_id, model_name, layer, values, dimension, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                embedding_id,
                sequence_id,
                model_name,
                layer,
                embedding.tolist(),
                len(embedding),
                datetime.now(),
            ],
        )
        return embedding_id

    def get_embeddings_by_sequence(
        self, sequence: str, model_name: Optional[str] = None, load_data: bool = False
    ) -> list[dict]:
        """Get embeddings for a sequence."""
        seq_hash = self._hash_sequence(sequence)

        sql = """
            SELECT e.*
            FROM embeddings e
            JOIN sequences s ON e.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params = [seq_hash]

        if model_name:
            sql += " AND e.model_name = ?"
            params.append(model_name)

        df = self.conn.execute(sql, params).df()
        results = df.to_dict("records")

        if load_data:
            for r in results:
                vals = r.get("values")
                if vals is not None:
                    # Inline storage: values is a Python list from DuckDB FLOAT[]
                    r["embedding"] = np.array(vals, dtype=np.float32)
                elif r.get("embedding_path"):
                    # Backward compat: old-style per-file Parquet
                    df_emb = pd.read_parquet(r["embedding_path"])
                    r["embedding"] = np.array(df_emb["values"].iloc[0])

        return results

    def create_pipeline_run(
        self, run_id: str, pipeline_type: str, config: dict, status: str = "running"
    ):
        """Create or update a pipeline run record (safe for resume runs)."""
        now = datetime.now()
        self.conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, pipeline_type, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at
        """,
            [
                run_id,
                pipeline_type,
                json.dumps(config),
                status,
                now,
                now,
            ],
        )

    def update_pipeline_run_status(self, run_id: str, status: str):
        """Update pipeline run status."""
        self.conn.execute(
            """
            UPDATE pipeline_runs
            SET status = ?, updated_at = ?
            WHERE run_id = ?
        """,
            [status, datetime.now(), run_id],
        )

    def mark_stage_complete(
        self,
        run_id: str,
        stage_name: str,
        stage_id: str,
        input_count: int,
        output_count: int,
        status: str = "completed",
    ):
        """Mark stage as complete (or failed/skipped)."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO stage_completions
            (stage_id, run_id, stage_name, status, input_count, output_count, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                stage_id,
                run_id,
                stage_name,
                status,
                input_count,
                output_count,
                datetime.now(),
            ],
        )

    def is_stage_complete(self, stage_id: str) -> bool:
        """Check if stage is complete."""
        result = self.conn.execute(
            """
            SELECT COUNT(*) FROM stage_completions
            WHERE stage_id = ? AND status = 'completed'
        """,
            [stage_id],
        ).fetchone()
        return result is not None and result[0] > 0

    def export_to_parquet(self, table_name: str, output_path: Union[str, Path]):
        """
        Export table to Parquet file (for sharing/archiving).

        Args:
            table_name: Table to export (sequences, predictions, etc.)
            output_path: Path to output Parquet file

        Example:
            >>> ds.export_to_parquet('sequences', 'sequences_backup.parquet')
        """
        self.conn.execute(
            f"""
            COPY {table_name} TO '{output_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """
        )

    def add_generation_metadata(
        self,
        sequence_id: int,
        model_name: str,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        max_length: Optional[int] = None,
        sampling_params: Optional[
            dict
        ] = None,  # extra params stored but not in dedicated columns
    ) -> int:
        """Store generation parameters for a sequence.

        Returns:
            metadata_id of the inserted row.
        """
        # Merge sampling_params into known fields where they overlap
        if sampling_params:
            top_k = top_k if top_k is not None else sampling_params.get("top_k")
            top_p = top_p if top_p is not None else sampling_params.get("top_p")

        metadata_id = self._generation_metadata_counter
        self.conn.execute(
            """
            INSERT INTO generation_metadata
            (metadata_id, sequence_id, model_name, temperature, top_k, top_p,
             num_return_sequences, do_sample, repetition_penalty, max_length, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                metadata_id,
                sequence_id,
                model_name,
                temperature,
                top_k,
                top_p,
                num_return_sequences,
                do_sample,
                repetition_penalty,
                max_length,
                datetime.now(),
            ],
        )
        self._generation_metadata_counter += 1
        return metadata_id

    def export_to_dataframe(
        self,
        include_sequences: bool = True,
        include_predictions: bool = True,
        include_generation_metadata: bool = False,
        prediction_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Export data to a flat DataFrame using a single DuckDB SQL query.

        Uses conditional aggregation (CASE WHEN pivot) — no per-type queries,
        no pandas merges, no full table loads.

        Args:
            include_sequences: Always True; includes sequence_id, sequence, length.
            include_predictions: Pivot prediction_type values into columns.
            include_generation_metadata: Join generation_metadata columns.
            prediction_types: Limit to specific prediction types (None = all).

        Returns:
            Wide-format DataFrame: one row per sequence, one column per prediction type.
        """
        # Determine which prediction types to pivot
        if include_predictions:
            if prediction_types is None:
                rows = self.conn.execute(
                    "SELECT DISTINCT prediction_type FROM predictions WHERE prediction_type IS NOT NULL"
                ).fetchall()
                prediction_types = [r[0] for r in rows]
            else:
                prediction_types = [pt for pt in prediction_types if pt]

        # Build CASE WHEN pivot for predictions
        pred_cols_sql = ""
        if include_predictions and prediction_types:
            cases = [
                "MAX(CASE WHEN p.prediction_type = '{}' THEN p.value END) AS \"{}\"".format(
                    pt.replace("'", "''"), pt.replace('"', '""')
                )
                for pt in prediction_types
            ]
            pred_cols_sql = ",\n            " + ",\n            ".join(cases)

        # Build generation metadata join
        gen_join_sql = ""
        gen_cols_sql = ""
        if include_generation_metadata:
            gen_join_sql = (
                "LEFT JOIN generation_metadata gm ON s.sequence_id = gm.sequence_id"
            )
            gen_cols_sql = """,
            gm.model_name AS gen_model_name,
            gm.temperature AS gen_temperature,
            gm.top_k AS gen_top_k,
            gm.top_p AS gen_top_p"""

        pred_join_sql = (
            "LEFT JOIN predictions p ON s.sequence_id = p.sequence_id"
            if include_predictions and prediction_types
            else ""
        )
        group_extra = (
            ", gm.model_name, gm.temperature, gm.top_k, gm.top_p"
            if include_generation_metadata
            else ""
        )

        # Discover all columns on sequences table for SELECT
        seq_cols = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'sequences' ORDER BY ordinal_position"
        ).fetchall()
        seq_col_names = [r[0] for r in seq_cols if r[0] != "created_at"]
        seq_select = ", ".join(f's."{c}"' for c in seq_col_names)
        seq_group = ", ".join(f's."{c}"' for c in seq_col_names)

        query = f"""
            SELECT
                {seq_select}{pred_cols_sql}{gen_cols_sql}
            FROM sequences s
            {pred_join_sql}
            {gen_join_sql}
            GROUP BY {seq_group}{group_extra}
        """
        return self.conn.execute(query).df()

    def get_embeddings_bulk(
        self,
        sequence_ids: list[int],
        model_name: Optional[str] = None,
    ) -> dict[int, np.ndarray]:
        """
        Fetch embeddings for multiple sequences in a single JOIN query.

        Replaces N individual get_embeddings_by_sequence() calls (O(n) queries → O(1)).

        Args:
            sequence_ids: List of sequence IDs to fetch.
            model_name: Optional model filter.

        Returns:
            Dict mapping sequence_id → numpy embedding array for sequences that have an embedding.
        """
        if not sequence_ids:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_emb_bulk_ids", ids_df)
        try:
            sql = """
                SELECT e.sequence_id, e.values, e.embedding_path
                FROM embeddings e
                INNER JOIN _emb_bulk_ids b ON e.sequence_id = b.sequence_id
            """
            params: list[Any] = []
            if model_name:
                sql += " AND e.model_name = ?"
                params.append(model_name)
            df = self.conn.execute(sql, params).df()
        finally:
            self.conn.unregister("_emb_bulk_ids")

        result: dict[int, np.ndarray] = {}
        for _, row in df.iterrows():
            vals = row.get("values")
            if vals is not None and not (isinstance(vals, float) and np.isnan(vals)):
                result[int(row["sequence_id"])] = np.array(vals, dtype=np.float32)
            elif row.get("embedding_path"):
                try:
                    df_emb = pd.read_parquet(row["embedding_path"])
                    result[int(row["sequence_id"])] = np.array(df_emb["values"].iloc[0])
                except Exception:
                    pass
        return result

    def get_uncached_sequence_ids(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> list[int]:
        """
        Return sequence_ids that do NOT yet have a given prediction (vectorized anti-join).

        Replaces N individual has_prediction() calls with a single SQL query.

        Args:
            sequence_ids: Candidate sequence IDs to check.
            prediction_type: Prediction type key.
            model_name: Model name.

        Returns:
            List of sequence_ids with no cached prediction.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_check_ids", ids_df)
        try:
            result = self.conn.execute(
                """
                SELECT c.sequence_id
                FROM _check_ids c
                LEFT JOIN predictions p
                    ON c.sequence_id = p.sequence_id
                    AND p.prediction_type = ?
                    AND p.model_name = ?
                WHERE p.prediction_id IS NULL
            """,
                [prediction_type, model_name],
            ).fetchall()
        finally:
            self.conn.unregister("_check_ids")
        return [r[0] for r in result]

    def get_predictions_bulk(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> pd.DataFrame:
        """
        Fetch predictions for multiple sequences in a single JOIN query.

        Replaces N individual get_predictions_by_sequence() calls.

        Returns:
            DataFrame with columns: sequence_id, value, metadata
        """
        if not sequence_ids:
            return pd.DataFrame(columns=["sequence_id", "value", "metadata"])
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_bulk_ids", ids_df)
        try:
            result = self.conn.execute(
                """
                SELECT p.sequence_id, p.value, p.metadata
                FROM predictions p
                INNER JOIN _bulk_ids b ON p.sequence_id = b.sequence_id
                WHERE p.prediction_type = ? AND p.model_name = ?
            """,
                [prediction_type, model_name],
            ).df()
        finally:
            self.conn.unregister("_bulk_ids")
        return result

    def count_matching_sequences(self, sequences: list[str]) -> int:
        """
        Count how many of the given sequences already exist in the datastore.

        Uses a single vectorized hash join instead of N individual lookups.
        Safe to call from any context — registers DataFrame explicitly.
        """
        if not sequences:
            return 0
        hashes_df = pd.DataFrame({"hash": [self._hash_sequence(s) for s in sequences]})
        self.conn.register("_count_hashes", hashes_df)
        try:
            result = self.conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM sequences s
                INNER JOIN _count_hashes h ON s.hash = h.hash
            """
            ).fetchone()
        finally:
            self.conn.unregister("_count_hashes")
        return result[0] if result else 0

    def set_pipeline_metadata(self, key: str, value: Any):
        """Upsert a key/value pair in the pipeline_metadata table."""
        self.conn.execute(
            """
            INSERT INTO pipeline_metadata (key, value, created_at) VALUES (?, ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """,
            [
                key,
                json.dumps(value) if not isinstance(value, str) else value,
                datetime.now(),
            ],
        )

    def get_pipeline_metadata(self, key: str) -> Optional[Any]:
        """Retrieve a value from pipeline_metadata by key."""
        result = self.conn.execute(
            "SELECT value FROM pipeline_metadata WHERE key = ?", [key]
        ).fetchone()
        if result is None:
            return None
        try:
            return json.loads(result[0])
        except (json.JSONDecodeError, TypeError):
            return result[0]

    # ------------------------------------------------------------------
    # Structure methods
    # ------------------------------------------------------------------

    def add_structure(
        self,
        sequence_id: int,
        model_name: str,
        structure_str: Optional[str] = None,
        format: str = "pdb",
        plddt_mean: Optional[float] = None,
        plddt: Optional[float] = None,  # alias for plddt_mean
    ) -> int:
        """Store a structure gzip-compressed as BLOB (~8-12x smaller than plain TEXT).

        Args:
            sequence_id: Sequence ID.
            model_name: Model that produced the structure (e.g. 'esmfold').
            structure_str: Full structure file content as a string.
            format: 'pdb' or 'cif' (default 'pdb').
            plddt_mean: Mean pLDDT score (optional).
            plddt: Alias for plddt_mean.

        Returns:
            structure_id of the inserted row.
        """
        actual_plddt = plddt_mean if plddt_mean is not None else plddt
        structure_id = self._structure_counter
        if structure_str is not None:
            compressed = gzip.compress(structure_str.encode("utf-8"), compresslevel=6)
        else:
            compressed = None
        self.conn.execute(
            """
            INSERT INTO structures
            (structure_id, sequence_id, model_name, format, structure_data, plddt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                structure_id,
                sequence_id,
                model_name,
                format,
                compressed,
                actual_plddt,
                datetime.now(),
            ],
        )
        self._structure_counter += 1
        return structure_id

    def get_structure(
        self,
        sequence_id: int,
        model_name: Optional[str] = None,
    ) -> Optional[dict]:
        """Fetch the most recent structure for a sequence, decompressing on read.

        Returns a dict with 'structure_str' key (always decompressed string) regardless
        of whether data was stored compressed (structure_data BLOB) or as legacy plain TEXT.

        Args:
            sequence_id: Sequence ID.
            model_name: Optional model filter.

        Returns:
            Dict with structure record, or None if not found.
        """
        sql = """
            SELECT structure_id, sequence_id, model_name, format,
                   structure_data, structure_str, structure_path, plddt, created_at
            FROM structures
            WHERE sequence_id = ?
        """
        params = [sequence_id]
        if model_name:
            sql += " AND model_name = ?"
            params.append(model_name)
        sql += " ORDER BY created_at DESC LIMIT 1"
        df = self.conn.execute(sql, params).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        # Decompress structure_data (new path) or fall back to structure_str (old data)
        data_blob = record.pop("structure_data", None)
        if data_blob is not None and not (
            isinstance(data_blob, float) and np.isnan(data_blob)
        ):
            try:
                blob_bytes = (
                    bytes(data_blob) if not isinstance(data_blob, bytes) else data_blob
                )
                record["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
            except Exception:
                pass  # fall back to whatever structure_str already holds
        return record

    def get_structures_bulk(self, sequence_ids: list[int]) -> pd.DataFrame:
        """Fetch structures for multiple sequences, decompressing structure content.

        Returns a DataFrame with a 'structure_str' column (always plain text) regardless
        of whether data was stored compressed (structure_data BLOB) or as legacy plain TEXT.

        Args:
            sequence_ids: List of sequence IDs to look up.

        Returns:
            DataFrame with one row per structure record.
        """
        if not sequence_ids:
            return pd.DataFrame(
                columns=[
                    "structure_id",
                    "sequence_id",
                    "model_name",
                    "format",
                    "structure_str",
                    "plddt",
                ]
            )
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_struct_ids", ids_df)
        try:
            result = self.conn.execute(
                """
                SELECT s.structure_id, s.sequence_id, s.model_name,
                       s.format, s.structure_data, s.structure_str, s.plddt
                FROM structures s
                INNER JOIN _struct_ids b ON s.sequence_id = b.sequence_id
            """
            ).df()
        finally:
            self.conn.unregister("_struct_ids")

        def _decompress_row(row):
            blob = row.get("structure_data")
            if blob is not None and not (isinstance(blob, float) and np.isnan(blob)):
                try:
                    blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                    return gzip.decompress(blob_bytes).decode("utf-8")
                except Exception:
                    pass
            return row.get("structure_str")

        result["structure_str"] = result.apply(_decompress_row, axis=1)
        result = result.drop(columns=["structure_data"], errors="ignore")
        return result

    # ------------------------------------------------------------------
    # Filter results methods (for resume support)
    # ------------------------------------------------------------------

    def save_filter_results(
        self,
        run_id: str,
        stage_name: str,
        passed_sequence_ids: list[int],
    ):
        """Record which sequence_ids passed a filter stage (for resume support).

        Args:
            run_id: Pipeline run ID.
            stage_name: Filter stage name.
            passed_sequence_ids: IDs of sequences that passed the filter.
        """
        if not passed_sequence_ids:
            return
        rows = []
        for seq_id in passed_sequence_ids:
            rows.append(
                {
                    "filter_id": self._filter_id_counter,
                    "run_id": run_id,
                    "stage_name": stage_name,
                    "sequence_id": seq_id,
                    "passed": True,
                    "created_at": datetime.now(),
                }
            )
            self._filter_id_counter += 1

        df = pd.DataFrame(rows)
        self.conn.register("_filter_batch", df)
        try:
            self.conn.execute(
                """
                INSERT INTO filter_results
                (filter_id, run_id, stage_name, sequence_id, passed, created_at)
                SELECT filter_id, run_id, stage_name, sequence_id, passed, created_at
                FROM _filter_batch
            """
            )
        finally:
            self.conn.unregister("_filter_batch")

    def get_filter_results(
        self,
        run_id: str,
        stage_name: str,
    ) -> list[int]:
        """Return sequence_ids that passed a given filter stage in a run.

        Args:
            run_id: Pipeline run ID.
            stage_name: Filter stage name.

        Returns:
            List of sequence_ids that passed the filter, or empty list if no data.
        """
        rows = self.conn.execute(
            """
            SELECT sequence_id
            FROM filter_results
            WHERE run_id = ? AND stage_name = ? AND passed = TRUE
        """,
            [run_id, stage_name],
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Convenience / compatibility methods
    # ------------------------------------------------------------------

    def get_sequence(self, sequence_id: int) -> Optional[str]:
        """Return the sequence string for a given sequence_id, or None if not found."""
        result = self.conn.execute(
            "SELECT sequence FROM sequences WHERE sequence_id = ?", [sequence_id]
        ).fetchone()
        return result[0] if result else None

    def get_all_sequences(self) -> pd.DataFrame:
        """Return all sequences as a DataFrame with sequence_id, sequence, length columns."""
        return self.conn.execute(
            "SELECT sequence_id, sequence, length FROM sequences ORDER BY sequence_id"
        ).df()

    def add_prediction_by_sequence(
        self,
        sequence: str,
        prediction_type: str,
        model_name: str,
        value: Optional[float],
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a prediction, creating the sequence if it doesn't exist.

        Returns:
            prediction_id of the inserted row.
        """
        seq_id = self.add_sequence(sequence)
        pred_id = self._prediction_counter  # captured before batch increments it
        self.add_predictions_batch(
            [
                {
                    "sequence_id": seq_id,
                    "prediction_type": prediction_type,
                    "model_name": model_name,
                    "value": value,
                    "metadata": metadata,
                }
            ]
        )
        return pred_id  # correct: add_predictions_batch increments after insert

    def get_predictions(
        self,
        sequence_id: int,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return predictions for a sequence_id as a DataFrame."""
        sql = "SELECT * FROM predictions WHERE sequence_id = ?"
        params: list[Any] = [sequence_id]
        if prediction_type:
            sql += " AND prediction_type = ?"
            params.append(prediction_type)
        if model_name:
            sql += " AND model_name = ?"
            params.append(model_name)
        return self.conn.execute(sql, params).df()

    def get_generation_metadata(self, sequence_id: int) -> list[dict]:
        """Return generation metadata records for a sequence_id."""
        df = self.conn.execute(
            "SELECT * FROM generation_metadata WHERE sequence_id = ?", [sequence_id]
        ).df()
        return df.to_dict("records")

    def get_structures_by_sequence(
        self,
        sequence: str,
        model_name: Optional[str] = None,
    ) -> list[dict]:
        """Return structure records for a sequence string (decompressed)."""
        seq_hash = self._hash_sequence(sequence)
        sql = """
            SELECT st.*
            FROM structures st
            JOIN sequences s ON st.sequence_id = s.sequence_id
            WHERE s.hash = ?
        """
        params: list[Any] = [seq_hash]
        if model_name:
            sql += " AND st.model_name = ?"
            params.append(model_name)
        df = self.conn.execute(sql, params).df()
        records = df.to_dict("records")
        for rec in records:
            blob = rec.pop("structure_data", None)
            if blob is not None and not (isinstance(blob, float) and np.isnan(blob)):
                try:
                    blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                    rec["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
                except Exception:
                    pass
        return records

    def get_structure_by_id(self, structure_id: int) -> Optional[dict]:
        """Return a structure record by its structure_id (primary key)."""
        df = self.conn.execute(
            "SELECT * FROM structures WHERE structure_id = ?", [structure_id]
        ).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        blob = record.pop("structure_data", None)
        if blob is not None and not (isinstance(blob, float) and np.isnan(blob)):
            try:
                blob_bytes = bytes(blob) if not isinstance(blob, bytes) else blob
                record["structure_str"] = gzip.decompress(blob_bytes).decode("utf-8")
            except Exception:
                pass
        return record

    def get_embedding(self, embedding_id: int) -> Optional[tuple]:
        """Return (metadata_dict, embedding_array) for a given embedding_id, or None."""
        df = self.conn.execute(
            "SELECT * FROM embeddings WHERE embedding_id = ?", [embedding_id]
        ).df()
        if df.empty:
            return None
        record = df.iloc[0].to_dict()
        vals = record.get("values")
        if vals is not None and not (isinstance(vals, float) and np.isnan(vals)):
            emb_array = np.array(vals, dtype=np.float32)
        else:
            emb_path = record.get("embedding_path")
            if emb_path:
                df_emb = pd.read_parquet(emb_path)
                emb_array = np.array(df_emb["values"].iloc[0])
            else:
                emb_array = None
        record.pop("values", None)
        return record, emb_array

    def get_pipeline_run(self, run_id: str) -> Optional[dict]:
        """Return pipeline run record as a dict, or None if not found."""
        df = self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", [run_id]
        ).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_stats(self) -> dict[str, int]:
        """Return row counts for the main tables."""
        return {
            "sequences": self.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[
                0
            ],
            "predictions": self.conn.execute(
                "SELECT COUNT(*) FROM predictions"
            ).fetchone()[0],
            "embeddings": self.conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0],
            "structures": self.conn.execute(
                "SELECT COUNT(*) FROM structures"
            ).fetchone()[0],
            "pipeline_runs": self.conn.execute(
                "SELECT COUNT(*) FROM pipeline_runs"
            ).fetchone()[0],
        }

    def export_to_csv(self, path: Union[str, Path], **kwargs) -> None:
        """Export all data to CSV (convenience wrapper around export_to_dataframe)."""
        df = self.export_to_dataframe(**kwargs)
        df.to_csv(str(path), index=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # WorkingSet support methods
    # ------------------------------------------------------------------

    def materialize_working_set(
        self,
        ws: WorkingSet,
        include_predictions: bool = True,
        prediction_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Materialize a WorkingSet into a DataFrame via a single DuckDB pivot query.

        Args:
            ws: WorkingSet containing the sequence IDs to materialize.
            include_predictions: If True, pivot prediction values into columns.
            prediction_types: Limit to specific types (None = all available).

        Returns:
            Wide-format DataFrame: one row per sequence, one column per prediction type.
        """
        if not ws:
            return pd.DataFrame(columns=["sequence_id", "sequence", "length"])

        ids_df = pd.DataFrame({"sequence_id": list(ws.sequence_ids)})
        self.conn.register("_ws_ids", ids_df)
        try:
            # Determine prediction types to pivot
            if include_predictions:
                if prediction_types is None:
                    rows = self.conn.execute(
                        """
                        SELECT DISTINCT p.prediction_type
                        FROM predictions p
                        INNER JOIN _ws_ids w ON p.sequence_id = w.sequence_id
                        WHERE p.prediction_type IS NOT NULL
                    """
                    ).fetchall()
                    prediction_types = [r[0] for r in rows]

            # Build CASE WHEN pivot
            pred_cols_sql = ""
            pred_join_sql = ""
            if include_predictions and prediction_types:
                cases = [
                    "MAX(CASE WHEN p.prediction_type = '{}' THEN p.value END) AS \"{}\"".format(
                        pt.replace("'", "''"), pt.replace('"', '""')
                    )
                    for pt in prediction_types
                ]
                pred_cols_sql = ",\n                " + ",\n                ".join(cases)
                pred_join_sql = (
                    "LEFT JOIN predictions p ON s.sequence_id = p.sequence_id"
                )

            # Discover all columns on sequences table for SELECT
            seq_cols = self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'sequences' ORDER BY ordinal_position"
            ).fetchall()
            seq_col_names = [r[0] for r in seq_cols if r[0] != "created_at"]
            seq_select = ", ".join(f's."{c}"' for c in seq_col_names)
            seq_group = ", ".join(f's."{c}"' for c in seq_col_names)

            query = f"""
                SELECT
                    {seq_select}{pred_cols_sql}
                FROM sequences s
                INNER JOIN _ws_ids w ON s.sequence_id = w.sequence_id
                {pred_join_sql}
                GROUP BY {seq_group}
            """
            return self.conn.execute(query).df()
        finally:
            self.conn.unregister("_ws_ids")

    def get_sequence_ids_with_prediction(
        self,
        sequence_ids: list[int],
        prediction_type: str,
        model_name: str,
    ) -> list[int]:
        """Return sequence_ids that DO have a given prediction (inverse of uncached check).

        Args:
            sequence_ids: Candidate sequence IDs.
            prediction_type: Prediction type key.
            model_name: Model name.

        Returns:
            List of sequence_ids that have a cached prediction.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_has_pred_ids", ids_df)
        try:
            result = self.conn.execute(
                """
                SELECT DISTINCT c.sequence_id
                FROM _has_pred_ids c
                INNER JOIN predictions p
                    ON c.sequence_id = p.sequence_id
                    AND p.prediction_type = ?
                    AND p.model_name = ?
            """,
                [prediction_type, model_name],
            ).fetchall()
        finally:
            self.conn.unregister("_has_pred_ids")
        return [r[0] for r in result]

    def get_sequences_for_ids(
        self,
        sequence_ids: list[int],
    ) -> list[tuple[int, str]]:
        """Fetch (sequence_id, sequence) pairs for the given IDs.

        Lightweight fetch for building API request items without materializing
        a full DataFrame.

        Args:
            sequence_ids: List of sequence IDs to look up.

        Returns:
            List of (sequence_id, sequence_string) tuples.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_seq_for_ids", ids_df)
        try:
            rows = self.conn.execute(
                """
                SELECT s.sequence_id, s.sequence
                FROM sequences s
                INNER JOIN _seq_for_ids w ON s.sequence_id = w.sequence_id
            """
            ).fetchall()
        finally:
            self.conn.unregister("_seq_for_ids")
        return [(r[0], r[1]) for r in rows]

    def execute_filter_sql(
        self,
        sequence_ids: list[int],
        sql_query: str,
    ) -> list[int]:
        """Execute a filter SQL query and return surviving sequence IDs.

        The *sql_query* must be a complete ``SELECT`` statement that returns
        ``sequence_id`` values.  It may reference the registered table
        ``_filter_ws`` (which contains the input *sequence_ids*) to scope
        results to the current working set.

        Args:
            sequence_ids: Input sequence IDs (registered as ``_filter_ws``).
            sql_query: Complete SQL SELECT returning ``sequence_id`` values,
                JOINed with ``_filter_ws`` to scope to the working set.

        Returns:
            List of sequence_ids that survive the filter.
        """
        if not sequence_ids:
            return []
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_filter_ws", ids_df)
        try:
            result = self.conn.execute(sql_query).fetchall()
        finally:
            self.conn.unregister("_filter_ws")
        return [r[0] for r in result]

    def store_sequence_attributes(
        self,
        seq_ids: list[int],
        attr_name: str,
        attr_values: list[str],
    ):
        """Persist a per-sequence attribute column (e.g. heavy_chain, light_chain).

        Args:
            seq_ids: Sequence IDs.
            attr_name: Attribute name (column name from the input DataFrame).
            attr_values: Corresponding values (one per sequence_id).
        """
        if not seq_ids:
            return
        df = pd.DataFrame(
            {
                "sequence_id": seq_ids,
                "attr_name": attr_name,
                "attr_value": [str(v) if v is not None else None for v in attr_values],
            }
        )
        self.conn.register("_attr_batch", df)
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO sequence_attributes (sequence_id, attr_name, attr_value)
                SELECT sequence_id, attr_name, attr_value FROM _attr_batch
            """
            )
        finally:
            self.conn.unregister("_attr_batch")

    def get_sequence_attributes_for_ids(
        self,
        sequence_ids: list[int],
        attr_names: list[str],
    ) -> dict[int, dict[str, str]]:
        """Retrieve per-sequence attributes, returning {seq_id: {attr: value}}.

        Args:
            sequence_ids: Sequence IDs to look up.
            attr_names: Attribute names to retrieve.

        Returns:
            Nested dict: ``{sequence_id: {attr_name: attr_value}}``.
        """
        if not sequence_ids or not attr_names:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_attr_ids", ids_df)
        try:
            placeholders = ", ".join(["?"] * len(attr_names))
            rows = self.conn.execute(
                f"""
                SELECT a.sequence_id, a.attr_name, a.attr_value
                FROM sequence_attributes a
                INNER JOIN _attr_ids w ON a.sequence_id = w.sequence_id
                WHERE a.attr_name IN ({placeholders})
            """,
                attr_names,
            ).fetchall()
        finally:
            self.conn.unregister("_attr_ids")

        result: dict[int, dict[str, str]] = {}
        for sid, name, val in rows:
            result.setdefault(int(sid), {})[name] = val
        return result

    # ------------------------------------------------------------------
    # Pipeline context methods (inter-stage key-value store)
    # ------------------------------------------------------------------

    def set_context(self, run_id: str, key: str, value: Any):
        """Store a key-value pair in the pipeline context table."""
        self.conn.execute(
            """
            INSERT INTO pipeline_context (run_id, key, value, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (run_id, key) DO UPDATE SET
                value = excluded.value, created_at = excluded.created_at
        """,
            [
                run_id,
                key,
                json.dumps(value) if not isinstance(value, str) else value,
                datetime.now(),
            ],
        )

    def get_context(self, run_id: str, key: str) -> Optional[Any]:
        """Retrieve a value from the pipeline context table."""
        result = self.conn.execute(
            "SELECT value FROM pipeline_context WHERE run_id = ? AND key = ?",
            [run_id, key],
        ).fetchone()
        if result is None:
            return None
        try:
            return json.loads(result[0])
        except (json.JSONDecodeError, TypeError):
            return result[0]

    def get_sequences_for_ids_with_columns(
        self,
        sequence_ids: list[int],
        columns: list[str],
    ) -> dict[int, dict[str, str]]:
        """Fetch column values from the sequences table for given IDs.

        This reads columns stored directly on the ``sequences`` table
        (via ``ensure_input_columns``), NOT from ``sequence_attributes``.

        Args:
            sequence_ids: Sequence IDs to look up.
            columns: Column names to fetch (must exist on the sequences table).

        Returns:
            ``{sequence_id: {col: value, ...}}``
        """
        if not sequence_ids or not columns:
            return {}
        ids_df = pd.DataFrame({"sequence_id": sequence_ids})
        self.conn.register("_col_ids", ids_df)
        try:
            col_list = ", ".join(f's."{c}"' for c in columns)
            rows = self.conn.execute(
                f"""
                SELECT s.sequence_id, {col_list}
                FROM sequences s
                INNER JOIN _col_ids w ON s.sequence_id = w.sequence_id
            """
            ).fetchall()
        finally:
            self.conn.unregister("_col_ids")

        result: dict[int, dict[str, str]] = {}
        for row in rows:
            sid = int(row[0])
            result[sid] = {col: row[i + 1] for i, col in enumerate(columns)}
        return result

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __repr__(self):
        """String representation."""
        seq_count = self.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        pred_count = self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        emb_count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        struct_count = self.conn.execute("SELECT COUNT(*) FROM structures").fetchone()[
            0
        ]

        return (
            f"DuckDBDataStore(db='{self.db_path}', "
            f"sequences={seq_count}, predictions={pred_count}, "
            f"embeddings={emb_count}, structures={struct_count})"
        )
