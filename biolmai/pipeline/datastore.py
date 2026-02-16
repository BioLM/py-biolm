"""
DataStore - SQLite-based storage for sequences, predictions, structures, and embeddings.

Provides efficient caching and retrieval with deduplication support.
"""

import sqlite3
import json
import hashlib
import os
import gzip
import pickle
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager


class DataStore:
    """
    SQLite-based storage for pipeline data with efficient caching.
    
    Features:
    - Automatic sequence deduplication
    - Efficient indexing for fast lookups
    - Lazy loading for large structures
    - Atomic writes
    - Compression for large objects
    
    Args:
        db_path: Path to SQLite database file
        data_dir: Directory for storing large binary data (structures, embeddings)
        create_if_missing: Create database and tables if they don't exist
    
    Example:
        >>> store = DataStore('pipeline.db', 'pipeline_data')
        >>> seq_id = store.add_sequence('MKTAYIA')
        >>> store.add_prediction(seq_id, 'stability', 'ddg_predictor', 2.5)
        >>> results = store.get_predictions_by_sequence('MKTAYIA')
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = 'pipeline.db',
        data_dir: Union[str, Path] = 'pipeline_data',
        create_if_missing: bool = True
    ):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        
        if create_if_missing:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        if create_if_missing:
            self._create_tables()
    
    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Sequences table (deduplicated)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence TEXT UNIQUE NOT NULL,
                length INTEGER,
                hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sequence ON sequences(sequence)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON sequences(hash)')
        
        # Generation metadata (flattened - no nested JSON)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_metadata (
                generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER,
                model_name TEXT,
                temperature REAL,
                top_k INTEGER,
                top_p REAL,
                num_return_sequences INTEGER,
                do_sample INTEGER,
                repetition_penalty REAL,
                max_length INTEGER,
                generation_timestamp TIMESTAMP,
                FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gen_seq ON generation_metadata(sequence_id)')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER,
                model_name TEXT,
                prediction_type TEXT,
                value REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_seq ON predictions(sequence_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_type ON predictions(prediction_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name)')
        
        # Structures table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER,
                model_name TEXT,
                format TEXT,
                structure_path TEXT,
                plddt REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_struct_seq ON structures(sequence_id)')
        
        # Embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER,
                model_name TEXT,
                layer INTEGER,
                embedding_path TEXT,
                dimension INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sequence_id) REFERENCES sequences(sequence_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emb_seq ON embeddings(sequence_id)')
        
        # Pipeline runs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                pipeline_type TEXT,
                config TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        # Stage completions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stage_completions (
                stage_id TEXT PRIMARY KEY,
                run_id TEXT,
                stage_name TEXT,
                status TEXT,
                input_count INTEGER,
                output_count INTEGER,
                completed_at TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
            )
        ''')
        
        self.conn.commit()
    
    @staticmethod
    def _hash_sequence(sequence: str) -> str:
        """Generate hash for sequence."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]
    
    # === Sequence Operations ===
    
    def add_sequence(self, sequence: str, return_existing: bool = True) -> int:
        """
        Add a sequence to the database (deduplicated).
        
        Args:
            sequence: Protein sequence string
            return_existing: If True, return existing sequence_id if duplicate
        
        Returns:
            sequence_id: Integer ID for the sequence
        """
        cursor = self.conn.cursor()
        seq_hash = self._hash_sequence(sequence)
        
        try:
            cursor.execute('''
                INSERT INTO sequences (sequence, length, hash)
                VALUES (?, ?, ?)
            ''', (sequence, len(sequence), seq_hash))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Sequence already exists
            if return_existing:
                cursor.execute('SELECT sequence_id FROM sequences WHERE sequence = ?', (sequence,))
                row = cursor.fetchone()
                return row[0] if row else None
            raise
    
    def add_sequences_batch(self, sequences: List[str]) -> List[int]:
        """
        Add multiple sequences in batch (deduplicated).
        
        Args:
            sequences: List of protein sequence strings
        
        Returns:
            List of sequence_ids
        """
        sequence_ids = []
        for seq in sequences:
            seq_id = self.add_sequence(seq)
            sequence_ids.append(seq_id)
        return sequence_ids
    
    def get_sequence_id(self, sequence: str) -> Optional[int]:
        """Get sequence_id for a sequence, or None if not found."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT sequence_id FROM sequences WHERE sequence = ?', (sequence,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_sequence(self, sequence_id: int) -> Optional[str]:
        """Get sequence string by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT sequence FROM sequences WHERE sequence_id = ?', (sequence_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_all_sequences(self) -> pd.DataFrame:
        """Get all sequences as a DataFrame."""
        return pd.read_sql_query('SELECT * FROM sequences', self.conn)
    
    # === Generation Metadata ===
    
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
        timestamp: Optional[datetime] = None,
        **extra_params
    ) -> int:
        """
        Add generation metadata for a sequence (flattened, not nested).
        
        Args:
            sequence_id: Sequence ID
            model_name: Model name
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_return_sequences: Number of sequences returned
            do_sample: Whether sampling was used
            repetition_penalty: Repetition penalty
            max_length: Maximum length
            timestamp: Generation timestamp
            **extra_params: Additional params (ignored, for compatibility)
        
        Returns:
            generation_id
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO generation_metadata 
            (sequence_id, model_name, temperature, top_k, top_p, num_return_sequences,
             do_sample, repetition_penalty, max_length, generation_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sequence_id,
            model_name,
            temperature,
            top_k,
            top_p,
            num_return_sequences,
            1 if do_sample else 0 if do_sample is not None else None,
            repetition_penalty,
            max_length,
            timestamp or datetime.now()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_generation_metadata(self, sequence_id: int) -> List[Dict]:
        """Get all generation metadata for a sequence."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM generation_metadata WHERE sequence_id = ?
        ''', (sequence_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    # === Prediction Operations ===
    
    def add_prediction(
        self,
        sequence_id: int,
        prediction_type: str,
        model_name: str,
        value: float,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a prediction result.
        
        Args:
            sequence_id: Sequence ID
            prediction_type: Type of prediction (e.g., 'stability', 'tm', 'solubility')
            model_name: Name of the model used
            value: Prediction value
            metadata: Optional metadata dict
        
        Returns:
            prediction_id
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (sequence_id, model_name, prediction_type, value, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            sequence_id,
            model_name,
            prediction_type,
            value,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_prediction_by_sequence(
        self,
        sequence: str,
        prediction_type: str,
        model_name: str,
        value: float,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add prediction by sequence string (auto-adds sequence if needed)."""
        seq_id = self.add_sequence(sequence)
        return self.add_prediction(seq_id, prediction_type, model_name, value, metadata)
    
    def get_predictions(
        self,
        sequence_id: Optional[int] = None,
        prediction_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query predictions with optional filters.
        
        Args:
            sequence_id: Filter by sequence ID
            prediction_type: Filter by prediction type
            model_name: Filter by model name
        
        Returns:
            DataFrame with predictions
        """
        query = 'SELECT * FROM predictions WHERE 1=1'
        params = []
        
        if sequence_id is not None:
            query += ' AND sequence_id = ?'
            params.append(sequence_id)
        if prediction_type is not None:
            query += ' AND prediction_type = ?'
            params.append(prediction_type)
        if model_name is not None:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_predictions_by_sequence(self, sequence: str) -> pd.DataFrame:
        """Get all predictions for a sequence."""
        seq_id = self.get_sequence_id(sequence)
        if seq_id is None:
            return pd.DataFrame()
        return self.get_predictions(sequence_id=seq_id)
    
    def has_prediction(
        self,
        sequence: str,
        prediction_type: str,
        model_name: str
    ) -> bool:
        """Check if a prediction exists for a sequence."""
        seq_id = self.get_sequence_id(sequence)
        if seq_id is None:
            return False
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE sequence_id = ? AND prediction_type = ? AND model_name = ?
        ''', (seq_id, prediction_type, model_name))
        count = cursor.fetchone()[0]
        return count > 0
    
    # === Structure Operations ===
    
    def add_structure(
        self,
        sequence_id: int,
        model_name: str,
        structure_data: str,
        format: str = 'pdb',
        plddt: Optional[float] = None,
        metadata: Optional[Dict] = None,
        compress: bool = True
    ) -> int:
        """
        Add a structure.
        
        Args:
            sequence_id: Sequence ID
            model_name: Structure prediction model name
            structure_data: Structure content (PDB/CIF string)
            format: Structure format ('pdb' or 'cif')
            plddt: Optional pLDDT score
            metadata: Optional metadata dict
            compress: Whether to compress structure data
        
        Returns:
            structure_id
        """
        # Save structure to disk
        struct_dir = self.data_dir / 'structures' / model_name
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'seq_{sequence_id}_{timestamp}.{format}'
        if compress:
            filename += '.gz'
        
        struct_path = struct_dir / filename
        
        if compress:
            with gzip.open(struct_path, 'wt') as f:
                f.write(structure_data)
        else:
            with open(struct_path, 'w') as f:
                f.write(structure_data)
        
        # Add to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO structures (sequence_id, model_name, format, structure_path, plddt, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            sequence_id,
            model_name,
            format,
            str(struct_path),
            plddt,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_structure(self, structure_id: int) -> Optional[Dict]:
        """
        Get structure by ID (lazy loads structure data).
        
        Returns:
            Dict with structure info including 'data' field with structure content
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM structures WHERE structure_id = ?', (structure_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        struct_dict = dict(row)
        
        # Load structure data from disk
        struct_path = Path(struct_dict['structure_path'])
        if struct_path.suffix == '.gz':
            with gzip.open(struct_path, 'rt') as f:
                struct_dict['data'] = f.read()
        else:
            with open(struct_path, 'r') as f:
                struct_dict['data'] = f.read()
        
        return struct_dict
    
    def get_structures_by_sequence(self, sequence: str, model_name: Optional[str] = None) -> List[Dict]:
        """Get all structures for a sequence."""
        seq_id = self.get_sequence_id(sequence)
        if seq_id is None:
            return []
        
        cursor = self.conn.cursor()
        if model_name:
            cursor.execute('''
                SELECT * FROM structures WHERE sequence_id = ? AND model_name = ?
            ''', (seq_id, model_name))
        else:
            cursor.execute('SELECT * FROM structures WHERE sequence_id = ?', (seq_id,))
        
        structures = []
        for row in cursor.fetchall():
            struct_dict = dict(row)
            # Note: Not loading structure data here for efficiency
            # Call get_structure() if you need the actual structure content
            structures.append(struct_dict)
        
        return structures
    
    # === Embedding Operations ===
    
    def add_embedding(
        self,
        sequence_id: int,
        model_name: str,
        embedding: np.ndarray,
        layer: Optional[int] = None,
        compress: bool = True
    ) -> int:
        """
        Add an embedding.
        
        Args:
            sequence_id: Sequence ID
            model_name: Embedding model name
            embedding: Numpy array of embeddings
            layer: Optional layer number
            compress: Whether to compress embedding data
        
        Returns:
            embedding_id
        """
        # Save embedding to disk
        emb_dir = self.data_dir / 'embeddings' / model_name
        emb_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        layer_str = f'_layer{layer}' if layer is not None else ''
        filename = f'seq_{sequence_id}{layer_str}_{timestamp}.npy'
        if compress:
            filename += '.gz'
        
        emb_path = emb_dir / filename
        
        if compress:
            with gzip.open(emb_path, 'wb') as f:
                np.save(f, embedding)
        else:
            np.save(emb_path, embedding)
        
        # Add to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO embeddings (sequence_id, model_name, layer, embedding_path, dimension)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            sequence_id,
            model_name,
            layer,
            str(emb_path),
            embedding.shape[-1]  # Last dimension is embedding size
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_embedding(self, embedding_id: int) -> Optional[Tuple[Dict, np.ndarray]]:
        """
        Get embedding by ID.
        
        Returns:
            Tuple of (metadata_dict, embedding_array)
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM embeddings WHERE embedding_id = ?', (embedding_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        emb_dict = dict(row)
        
        # Load embedding from disk
        emb_path = Path(emb_dict['embedding_path'])
        if emb_path.suffix == '.gz':
            with gzip.open(emb_path, 'rb') as f:
                embedding = np.load(f)
        else:
            embedding = np.load(emb_path)
        
        return emb_dict, embedding
    
    def get_embeddings_by_sequence(
        self,
        sequence: str,
        model_name: Optional[str] = None,
        layer: Optional[int] = None,
        load_data: bool = False
    ) -> List[Union[Dict, Tuple[Dict, np.ndarray]]]:
        """
        Get embeddings for a sequence.
        
        Args:
            sequence: Sequence string
            model_name: Optional filter by model
            layer: Optional filter by layer
            load_data: If True, load embedding arrays; if False, return metadata only
        
        Returns:
            List of embedding dicts or (dict, array) tuples if load_data=True
        """
        seq_id = self.get_sequence_id(sequence)
        if seq_id is None:
            return []
        
        cursor = self.conn.cursor()
        query = 'SELECT * FROM embeddings WHERE sequence_id = ?'
        params = [seq_id]
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        if layer is not None:
            query += ' AND layer = ?'
            params.append(layer)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            emb_dict = dict(row)
            
            if load_data:
                emb_path = Path(emb_dict['embedding_path'])
                if emb_path.suffix == '.gz':
                    with gzip.open(emb_path, 'rb') as f:
                        embedding = np.load(f)
                else:
                    embedding = np.load(emb_path)
                results.append((emb_dict, embedding))
            else:
                results.append(emb_dict)
        
        return results
    
    # === Pipeline Run Operations ===
    
    def create_pipeline_run(
        self,
        run_id: str,
        pipeline_type: str,
        config: Dict,
        status: str = 'running'
    ):
        """Create a new pipeline run record."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO pipeline_runs (run_id, pipeline_type, config, status, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, pipeline_type, json.dumps(config), status, datetime.now()))
        self.conn.commit()
    
    def update_pipeline_run_status(self, run_id: str, status: str):
        """Update pipeline run status."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE pipeline_runs SET status = ?, updated_at = ? WHERE run_id = ?
        ''', (status, datetime.now(), run_id))
        self.conn.commit()
    
    def get_pipeline_run(self, run_id: str) -> Optional[Dict]:
        """Get pipeline run info."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM pipeline_runs WHERE run_id = ?', (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def mark_stage_complete(
        self,
        stage_id: str,
        run_id: str,
        stage_name: str,
        input_count: int,
        output_count: int,
        status: str = 'completed'
    ):
        """Mark a pipeline stage as complete."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stage_completions 
            (stage_id, run_id, stage_name, status, input_count, output_count, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (stage_id, run_id, stage_name, status, input_count, output_count, datetime.now()))
        self.conn.commit()
    
    def is_stage_complete(self, stage_id: str) -> bool:
        """Check if a stage is marked complete."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT status FROM stage_completions WHERE stage_id = ? AND status = 'completed'
        ''', (stage_id,))
        return cursor.fetchone() is not None
    
    # === Export Operations ===
    
    def export_to_dataframe(
        self,
        include_sequences: bool = True,
        include_predictions: bool = True,
        include_generation_metadata: bool = False,
        prediction_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Export data to a pandas DataFrame.
        
        Args:
            include_sequences: Include sequence info
            include_predictions: Include predictions
            include_generation_metadata: Include generation metadata
            prediction_types: Optional list of prediction types to include
        
        Returns:
            DataFrame with merged data
        """
        df = None
        
        if include_sequences:
            df = self.get_all_sequences()
        
        if include_predictions and df is not None:
            # Get predictions and pivot
            pred_query = 'SELECT sequence_id, prediction_type, model_name, value FROM predictions'
            if prediction_types:
                placeholders = ','.join('?' * len(prediction_types))
                pred_query += f' WHERE prediction_type IN ({placeholders})'
                df_pred = pd.read_sql_query(pred_query, self.conn, params=prediction_types)
            else:
                df_pred = pd.read_sql_query(pred_query, self.conn)
            
            # Pivot predictions to wide format
            if not df_pred.empty:
                # Create column name as prediction_type_model_name
                df_pred['col_name'] = df_pred['prediction_type'] + '_' + df_pred['model_name']
                df_pivot = df_pred.pivot_table(
                    index='sequence_id',
                    columns='col_name',
                    values='value',
                    aggfunc='first'
                ).reset_index()
                
                df = df.merge(df_pivot, on='sequence_id', how='left')
        
        if include_generation_metadata and df is not None:
            # Get all generation metadata columns (flattened)
            gen_query = '''
                SELECT sequence_id, model_name as gen_model, temperature, 
                       top_k, top_p, num_return_sequences, do_sample,
                       repetition_penalty, max_length
                FROM generation_metadata
            '''
            df_gen = pd.read_sql_query(gen_query, self.conn)
            
            if not df_gen.empty:
                df = df.merge(df_gen, on='sequence_id', how='left')
        
        return df if df is not None else pd.DataFrame()
    
    def export_to_csv(self, output_path: str, **kwargs):
        """Export data to CSV."""
        df = self.export_to_dataframe(**kwargs)
        df.to_csv(output_path, index=False)
        print(f'Exported {len(df)} sequences to {output_path}')
    
    # === Utility Methods ===
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor."""
        self.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        for table in ['sequences', 'predictions', 'structures', 'embeddings', 'pipeline_runs']:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[table] = cursor.fetchone()[0]
        
        return stats
    
    def __repr__(self):
        stats = self.get_stats()
        return (
            f"DataStore(db='{self.db_path}', "
            f"sequences={stats['sequences']}, "
            f"predictions={stats['predictions']}, "
            f"structures={stats['structures']}, "
            f"embeddings={stats['embeddings']})"
        )
