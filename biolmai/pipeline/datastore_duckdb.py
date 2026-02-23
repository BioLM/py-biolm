"""
DuckDB-based DataStore with Parquet backend for efficient large-scale data management.

Key features:
- Columnar storage (Parquet)
- Out-of-core queries (bigger than RAM)
- Vectorized anti-join deduplication
- 5-50× faster than pandas for aggregations
- Diff-mode friendly batch inserts
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import hashlib
import gzip
import json


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
        >>> ds.add_prediction(seq_id, "tm", "temberture", 65.5)
        >>> 
        >>> # Efficient query - no memory explosion
        >>> high_tm = ds.query("SELECT * FROM predictions WHERE value > 60")
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = 'pipeline.duckdb',
        data_dir: Union[str, Path] = './pipeline_data'
    ):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Parquet directories
        self.sequences_dir = self.data_dir / 'sequences'
        self.predictions_dir = self.data_dir / 'predictions'
        self.embeddings_dir = self.data_dir / 'embeddings'
        self.structures_dir = self.data_dir / 'structures'
        
        for d in [self.sequences_dir, self.predictions_dir, 
                  self.embeddings_dir, self.structures_dir]:
            d.mkdir(exist_ok=True)
        
        # Connect to DuckDB (persistent)
        self.conn = duckdb.connect(str(self.db_path))
        
        # Initialize schema
        self._init_schema()
    
    def _init_schema(self):
        """Initialize DuckDB tables backed by Parquet files."""
        
        # Sequences table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id INTEGER PRIMARY KEY,
                sequence VARCHAR NOT NULL,
                length INTEGER,
                hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create unique index for deduplication
        self.conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sequence_hash 
            ON sequences(hash)
        """)
        
        # Predictions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                prediction_type VARCHAR,
                value DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for fast queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_seq 
            ON predictions(sequence_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_type 
            ON predictions(prediction_type, model_name)
        """)
        
        # Embeddings metadata table (actual arrays in Parquet files)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                layer INTEGER,
                embedding_path VARCHAR,
                dimension INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Structures metadata table (actual files on disk)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY,
                sequence_id INTEGER,
                model_name VARCHAR,
                format VARCHAR,
                structure_path VARCHAR,
                plddt DOUBLE,
                metadata VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pipeline runs table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id VARCHAR PRIMARY KEY,
                pipeline_type VARCHAR,
                config VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        # Stage completions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stage_completions (
                stage_id VARCHAR PRIMARY KEY,
                run_id VARCHAR,
                stage_name VARCHAR,
                status VARCHAR,
                input_count INTEGER,
                output_count INTEGER,
                completed_at TIMESTAMP
            )
        """)

        # Pipeline metadata (clustering results, stage diagnostics, etc.)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Generation metadata (parameters used to produce each generated sequence)
        self.conn.execute("""
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
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gen_meta_seq
            ON generation_metadata(sequence_id)
        """)

        # Initialize sequence counter
        result = self.conn.execute("SELECT MAX(sequence_id) FROM sequences").fetchone()
        self._sequence_counter = (result[0] or 0) + 1
        
        result = self.conn.execute("SELECT MAX(prediction_id) FROM predictions").fetchone()
        self._prediction_counter = (result[0] or 0) + 1
        
        result = self.conn.execute("SELECT MAX(embedding_id) FROM embeddings").fetchone()
        self._embedding_counter = (result[0] or 0) + 1
        
        result = self.conn.execute("SELECT MAX(structure_id) FROM structures").fetchone()
        self._structure_counter = (result[0] or 0) + 1

        result = self.conn.execute("SELECT MAX(metadata_id) FROM generation_metadata").fetchone()
        self._generation_metadata_counter = (result[0] or 0) + 1
    
    @staticmethod
    def _hash_sequence(sequence: str) -> str:
        """Create hash of sequence for deduplication."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]
    
    def add_sequences_batch(
        self,
        sequences: List[str],
        deduplicate: bool = True
    ) -> List[int]:
        """
        Add multiple sequences efficiently using anti-join deduplication.
        
        This is the RECOMMENDED way to add sequences - vectorized and fast!
        
        Args:
            sequences: List of sequence strings
            deduplicate: Use anti-join to skip existing sequences
        
        Returns:
            List of sequence_ids (new and existing)
        """
        if not sequences:
            return []
        
        # Create DataFrame for incoming batch
        df_new = pd.DataFrame({
            'sequence': sequences,
            'length': [len(s) for s in sequences],
            'hash': [self._hash_sequence(s) for s in sequences]
        })
        
        # Remove duplicates within the batch
        df_new = df_new.drop_duplicates(subset=['hash'], keep='first')
        
        if deduplicate:
            # 🔥 Anti-join: find sequences NOT already in database
            # This is vectorized and super fast!
            df_to_insert = self.conn.execute("""
                SELECT s.*
                FROM df_new s
                LEFT JOIN sequences t
                ON s.hash = t.hash
                WHERE t.hash IS NULL
            """).df()
        else:
            df_to_insert = df_new
        
        # Assign IDs
        if len(df_to_insert) > 0:
            start_id = self._sequence_counter
            df_to_insert['sequence_id'] = range(start_id, start_id + len(df_to_insert))
            df_to_insert['created_at'] = datetime.now()
            
            # Batch insert (vectorized!) - specify column order explicitly
            self.conn.execute("""
                INSERT INTO sequences (sequence_id, sequence, length, hash, created_at)
                SELECT sequence_id, sequence, length, hash, created_at FROM df_to_insert
            """)
            
            self._sequence_counter += len(df_to_insert)
        
        # Get all IDs (including existing)
        df_result = self.conn.execute("""
            SELECT hash, sequence_id
            FROM sequences
            WHERE hash IN (SELECT hash FROM df_new)
        """).df()
        
        # Map back to original order
        hash_to_id = dict(zip(df_result['hash'], df_result['sequence_id']))
        return [hash_to_id[self._hash_sequence(seq)] for seq in sequences]
    
    def add_sequence(self, sequence: str) -> int:
        """Add single sequence (convenience wrapper)."""
        return self.add_sequences_batch([sequence])[0]
    
    def add_predictions_batch(
        self,
        data: List[Dict[str, Any]]
    ):
        """
        Batch add predictions efficiently.
        
        Args:
            data: List of dicts with keys: sequence_id, prediction_type, 
                  model_name, value, metadata (optional)
        
        Example:
            >>> ds.add_predictions_batch([
            ...     {'sequence_id': 1, 'prediction_type': 'tm', 
            ...      'model_name': 'temberture', 'value': 65.5},
            ...     {'sequence_id': 2, 'prediction_type': 'tm', 
            ...      'model_name': 'temberture', 'value': 70.2},
            ... ])
        """
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Assign IDs
        start_id = self._prediction_counter
        df['prediction_id'] = range(start_id, start_id + len(df))
        df['created_at'] = datetime.now()
        
        # Ensure metadata is JSON string
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
        else:
            df['metadata'] = None
        
        # Batch insert - specify column order explicitly
        self.conn.execute("""
            INSERT INTO predictions (prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at)
            SELECT prediction_id, sequence_id, model_name, prediction_type, value, metadata, created_at FROM df
        """)
        self._prediction_counter += len(df)
    
    def add_prediction(
        self,
        sequence_id: int,
        prediction_type: str,
        model_name: str,
        value: Optional[float],
        metadata: Optional[Dict] = None
    ):
        """Add single prediction (convenience wrapper)."""
        self.add_predictions_batch([{
            'sequence_id': sequence_id,
            'prediction_type': prediction_type,
            'model_name': model_name,
            'value': value,
            'metadata': metadata
        }])
    
    def has_prediction(
        self,
        sequence: str,
        prediction_type: str,
        model_name: str
    ) -> bool:
        """Check if prediction exists for sequence."""
        seq_hash = self._hash_sequence(sequence)
        result = self.conn.execute("""
            SELECT COUNT(*) 
            FROM predictions p
            JOIN sequences s ON p.sequence_id = s.sequence_id
            WHERE s.hash = ? 
            AND p.prediction_type = ? 
            AND p.model_name = ?
        """, [seq_hash, prediction_type, model_name]).fetchone()
        
        return result[0] > 0
    
    def query(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
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
        model_name: Optional[str] = None
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
            "SELECT sequence_id FROM sequences WHERE hash = ?",
            [seq_hash]
        ).fetchone()
        return result[0] if result else None
    
    def add_embedding(
        self,
        sequence_id: int,
        model_name: str,
        embedding: np.ndarray,
        layer: Optional[int] = None
    ):
        """
        Add embedding (stored in Parquet for efficiency).
        
        Args:
            sequence_id: Sequence ID
            model_name: Model name
            embedding: Numpy array
            layer: Optional layer number
        """
        # Store embedding in Parquet file
        embedding_id = self._embedding_counter
        self._embedding_counter += 1
        
        embedding_path = self.embeddings_dir / f"emb_{embedding_id}.parquet"
        
        # Store as Parquet (columnar, compressed)
        df_emb = pd.DataFrame({'values': [embedding.tolist()]})
        df_emb.to_parquet(embedding_path, compression='snappy')
        
        # Store metadata in DuckDB
        self.conn.execute("""
            INSERT INTO embeddings 
            (embedding_id, sequence_id, model_name, layer, embedding_path, dimension, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            embedding_id,
            sequence_id,
            model_name,
            layer,
            str(embedding_path),
            len(embedding),
            datetime.now()
        ])
    
    def get_embeddings_by_sequence(
        self,
        sequence: str,
        model_name: Optional[str] = None,
        load_data: bool = False
    ) -> List[Dict]:
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
        results = df.to_dict('records')
        
        # Optionally load embedding data from Parquet
        if load_data:
            for r in results:
                df_emb = pd.read_parquet(r['embedding_path'])
                r['embedding'] = np.array(df_emb['values'].iloc[0])
        
        return results
    
    def create_pipeline_run(
        self,
        run_id: str,
        pipeline_type: str,
        config: Dict,
        status: str = 'running'
    ):
        """Create pipeline run record."""
        self.conn.execute("""
            INSERT INTO pipeline_runs (run_id, pipeline_type, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            pipeline_type,
            json.dumps(config),
            status,
            datetime.now(),
            datetime.now()
        ])
    
    def update_pipeline_run_status(self, run_id: str, status: str):
        """Update pipeline run status."""
        self.conn.execute("""
            UPDATE pipeline_runs 
            SET status = ?, updated_at = ?
            WHERE run_id = ?
        """, [status, datetime.now(), run_id])
    
    def mark_stage_complete(
        self,
        run_id: str,
        stage_name: str,
        stage_id: str,
        input_count: int,
        output_count: int,
        status: str = 'completed'
    ):
        """Mark stage as complete (or failed/skipped)."""
        self.conn.execute("""
            INSERT OR REPLACE INTO stage_completions
            (stage_id, run_id, stage_name, status, input_count, output_count, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [stage_id, run_id, stage_name, status, input_count, output_count, datetime.now()])
    
    def is_stage_complete(self, stage_id: str) -> bool:
        """Check if stage is complete."""
        result = self.conn.execute("""
            SELECT COUNT(*) FROM stage_completions 
            WHERE stage_id = ? AND status = 'completed'
        """, [stage_id]).fetchone()
        return result[0] > 0
    
    def export_to_parquet(
        self,
        table_name: str,
        output_path: Union[str, Path]
    ):
        """
        Export table to Parquet file (for sharing/archiving).
        
        Args:
            table_name: Table to export (sequences, predictions, etc.)
            output_path: Path to output Parquet file
        
        Example:
            >>> ds.export_to_parquet('sequences', 'sequences_backup.parquet')
        """
        self.conn.execute(f"""
            COPY {table_name} TO '{output_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
    
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
    ):
        """Store generation parameters for a sequence."""
        metadata_id = self._generation_metadata_counter
        self._generation_metadata_counter += 1
        self.conn.execute("""
            INSERT INTO generation_metadata
            (metadata_id, sequence_id, model_name, temperature, top_k, top_p,
             num_return_sequences, do_sample, repetition_penalty, max_length, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            metadata_id, sequence_id, model_name, temperature, top_k, top_p,
            num_return_sequences, do_sample, repetition_penalty, max_length, datetime.now()
        ])

    def export_to_dataframe(
        self,
        include_sequences: bool = True,
        include_predictions: bool = True,
        include_generation_metadata: bool = False,
        prediction_types: Optional[List[str]] = None,
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
                f"MAX(CASE WHEN p.prediction_type = '{pt}' THEN p.value END) AS \"{pt}\""
                for pt in prediction_types
            ]
            pred_cols_sql = ",\n            " + ",\n            ".join(cases)

        # Build generation metadata join
        gen_join_sql = ""
        gen_cols_sql = ""
        if include_generation_metadata:
            gen_join_sql = "LEFT JOIN generation_metadata gm ON s.sequence_id = gm.sequence_id"
            gen_cols_sql = """,
            gm.model_name AS gen_model_name,
            gm.temperature AS gen_temperature,
            gm.top_k AS gen_top_k,
            gm.top_p AS gen_top_p"""

        pred_join_sql = "LEFT JOIN predictions p ON s.sequence_id = p.sequence_id" if include_predictions and prediction_types else ""
        group_extra = ", gm.model_name, gm.temperature, gm.top_k, gm.top_p" if include_generation_metadata else ""

        query = f"""
            SELECT
                s.sequence_id,
                s.sequence,
                s.length{pred_cols_sql}{gen_cols_sql}
            FROM sequences s
            {pred_join_sql}
            {gen_join_sql}
            GROUP BY s.sequence_id, s.sequence, s.length{group_extra}
        """
        return self.conn.execute(query).df()

    def get_uncached_sequence_ids(
        self,
        sequence_ids: List[int],
        prediction_type: str,
        model_name: str,
    ) -> List[int]:
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
        ids_df = pd.DataFrame({'sequence_id': sequence_ids})
        self.conn.register('_check_ids', ids_df)
        result = self.conn.execute("""
            SELECT c.sequence_id
            FROM _check_ids c
            LEFT JOIN predictions p
                ON c.sequence_id = p.sequence_id
                AND p.prediction_type = ?
                AND p.model_name = ?
            WHERE p.prediction_id IS NULL
        """, [prediction_type, model_name]).fetchall()
        self.conn.unregister('_check_ids')
        return [r[0] for r in result]

    def get_predictions_bulk(
        self,
        sequence_ids: List[int],
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
            return pd.DataFrame(columns=['sequence_id', 'value', 'metadata'])
        ids_df = pd.DataFrame({'sequence_id': sequence_ids})
        self.conn.register('_bulk_ids', ids_df)
        result = self.conn.execute("""
            SELECT p.sequence_id, p.value, p.metadata
            FROM predictions p
            INNER JOIN _bulk_ids b ON p.sequence_id = b.sequence_id
            WHERE p.prediction_type = ? AND p.model_name = ?
        """, [prediction_type, model_name]).df()
        self.conn.unregister('_bulk_ids')
        return result

    def count_matching_sequences(self, sequences: List[str]) -> int:
        """
        Count how many of the given sequences already exist in the datastore.

        Uses a single vectorized hash join instead of N individual lookups.
        Safe to call from any context — registers DataFrame explicitly.
        """
        if not sequences:
            return 0
        hashes_df = pd.DataFrame({'hash': [self._hash_sequence(s) for s in sequences]})
        self.conn.register('_count_hashes', hashes_df)
        result = self.conn.execute("""
            SELECT COUNT(*) AS cnt
            FROM sequences s
            INNER JOIN _count_hashes h ON s.hash = h.hash
        """).fetchone()
        self.conn.unregister('_count_hashes')
        return result[0] if result else 0

    def set_pipeline_metadata(self, key: str, value: Any):
        """Upsert a key/value pair in the pipeline_metadata table."""
        self.conn.execute("""
            INSERT INTO pipeline_metadata (key, value, created_at) VALUES (?, ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """, [key, json.dumps(value) if not isinstance(value, str) else value, datetime.now()])

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

    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __repr__(self):
        """String representation."""
        seq_count = self.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
        pred_count = self.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        emb_count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        struct_count = self.conn.execute("SELECT COUNT(*) FROM structures").fetchone()[0]
        
        return (
            f"DuckDBDataStore(db='{self.db_path}', "
            f"sequences={seq_count}, predictions={pred_count}, "
            f"embeddings={emb_count}, structures={struct_count})"
        )
