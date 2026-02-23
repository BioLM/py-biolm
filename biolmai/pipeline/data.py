"""
Data-driven pipeline implementations.

DataPipeline: Load sequences from files/lists and run predictions
SingleStepPipeline: Simplified single-step prediction pipeline
"""

import json
import pandas as pd
import numpy as np
import asyncio
from typing import Union, List, Optional, Dict, Any, Tuple
from pathlib import Path

from biolmai.pipeline.base import BasePipeline, Stage, StageResult
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolmai.pipeline.filters import BaseFilter
from biolmai.client import BioLMApiClient  # Use async client directly


class PredictionStage(Stage):
    """
    Generic prediction stage using BioLM API.
    
    Args:
        name: Stage name
        model_name: BioLM model name (e.g., 'esmfold', 'esm2', 'temberture')
        action: API action ('predict', 'encode', 'score')
        prediction_type: Type of prediction for caching (e.g., 'structure', 'stability', 'embedding')
        params: Optional parameters for the API call
        batch_size: Batch size for API calls
        max_concurrent: Maximum concurrent batches
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        action: str = 'predict',
        prediction_type: Optional[str] = None,
        params: Optional[Dict] = None,
        batch_size: int = 32,
        max_concurrent: int = 5,
        **kwargs
    ):
        # Extract skip_on_error before passing to parent
        skip_on_error = kwargs.pop('skip_on_error', False)
        
        super().__init__(
            name=name,
            cache_key=prediction_type or f"{model_name}_{action}",
            model_name=model_name,
            max_concurrent=max_concurrent,
            **kwargs
        )
        self.action = action
        self.prediction_type = prediction_type or f"{model_name}_{action}"
        self.params = params or {}
        self.batch_size = batch_size
        self.skip_on_error = skip_on_error
        # Reuse API client across calls for connection pooling
        self._api_client = None
    
    async def process_streaming(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ):
        """
        Process sequences and yield results as batches complete (streaming).
        
        Yields DataFrames as API batches complete instead of waiting for all results.
        This allows downstream stages to start processing immediately.
        """
        from typing import AsyncIterator
        
        start_count = len(df)
        
        # Check cache for existing predictions
        uncached_mask = df.apply(
            lambda row: not datastore.has_prediction(
                row['sequence'],
                self.prediction_type,
                self.model_name
            ),
            axis=1
        )
        
        df_uncached = df[uncached_mask].copy()
        cached_count = start_count - len(df_uncached)
        
        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)} (streaming)")
        
        # Yield cached results first
        if cached_count > 0:
            df_cached = df[~uncached_mask].copy()
            # Load predictions for cached sequences
            for idx, row in df_cached.iterrows():
                pred = datastore.get_predictions_by_sequence(
                    row['sequence'],
                    self.prediction_type,
                    self.model_name
                )
                if pred:
                    df_cached.at[idx, self.prediction_type] = pred[0]['value']
            yield df_cached
        
        if len(df_uncached) == 0:
            return
        
        # Create or reuse async API client
        if self._api_client is None:
            self._api_client = BioLMApiClient(
                self.model_name,
                semaphore=self._semaphore,
                retry_error_batches=True
            )
        api = self._api_client
        
        # Batch sequences for API calls
        sequences = df_uncached['sequence'].tolist()
        batch_size = 32  # API batch size
        
        # Create tasks for all batches and start them immediately
        pending_tasks = {}  # task -> (batch_seqs, batch_indices)
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_indices = df_uncached.index[i:i+batch_size]
            items = [{'sequence': seq} for seq in batch_seqs]
            
            # Create and start task immediately
            if self.action == 'encode':
                task = asyncio.create_task(api.encode(items=items, params=self.params))
            else:
                task = asyncio.create_task(api.predict(items=items, params=self.params))
            
            pending_tasks[task] = (batch_seqs, batch_indices)
        
        # Process batches as they complete (true streaming!)
        remaining_tasks = set(pending_tasks.keys())
        
        while remaining_tasks:
            # Wait for next batch to complete
            done, remaining_tasks = await asyncio.wait(
                remaining_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for completed_task in done:
                try:
                    results = await completed_task
                    batch_seqs, batch_indices = pending_tasks[completed_task]
                    
                    # Create DataFrame for this batch
                    batch_df = pd.DataFrame({
                        'sequence': batch_seqs
                    }, index=batch_indices)
                    
                    # Store results and add to DataFrame
                    for seq, result, idx in zip(batch_seqs, results, batch_indices):
                        seq_id = datastore.add_sequence(seq)
                        
                        if self.action == 'predict':
                            # Extract prediction value
                            if isinstance(result, dict):
                                value = self._extract_prediction_value(result)
                            else:
                                value = float(result) if result is not None else None
                            
                            if value is not None:
                                datastore.add_prediction(
                                    seq_id,
                                    self.prediction_type,
                                    self.model_name,
                                    value,
                                    metadata={'params': self.params, 'result': result}
                                )
                                batch_df.at[idx, self.prediction_type] = value
                        
                        elif self.action == 'encode':
                            # Store embeddings (simplified)
                            if isinstance(result, dict) and 'embedding' in result:
                                embedding = np.array(result['embedding'])
                                datastore.add_embedding(seq_id, self.model_name, embedding)
                    
                    # Yield batch immediately!
                    yield batch_df
                    
                except Exception as e:
                    if self.skip_on_error:
                        # Mark batch sequences as failed in cache
                        batch_seqs, batch_indices = pending_tasks[completed_task]
                        print(f"  Error processing batch (skipped): {e}")
                        
                        for seq in batch_seqs:
                            seq_id = datastore.add_sequence(seq)
                            # Store failed prediction with metadata
                            datastore.add_prediction(
                                seq_id,
                                self.prediction_type,
                                self.model_name,
                                value=None,
                                metadata={
                                    'status': 'failed',
                                    'error': str(e),
                                    'params': self.params
                                }
                            )
                        # Don't yield failed batch - sequences are filtered out
                    else:
                        # Propagate error
                        print(f"  Error processing batch: {e}")
                        raise
    
    def _extract_prediction_value(self, result: dict) -> Optional[float]:
        """Extract primary numeric prediction value from API result dict."""
        for key in ('melting_temperature', 'solubility_score', 'prediction', 'score', 'value'):
            if key in result and isinstance(result[key], (int, float)):
                return float(result[key])
        return next((float(v) for v in result.values() if isinstance(v, (int, float))), None)

    def _store_embeddings(self, datastore: DataStore, seq_id: int, result: Any):
        """Store embedding(s) from a single API result dict into the datastore."""
        if not isinstance(result, dict):
            return
        if 'embedding' in result:
            embedding = np.array(result['embedding'])
            if embedding.size > 0:
                datastore.add_embedding(seq_id, self.model_name, embedding)
        elif 'embeddings' in result:
            embs = result['embeddings']
            if isinstance(embs, list):
                for emb_item in embs:
                    if isinstance(emb_item, dict):
                        layer_num = emb_item.get('layer')
                        emb_data = emb_item.get('embedding')
                        if emb_data is not None:
                            embedding = np.array(emb_data)
                            if embedding.size > 0:
                                datastore.add_embedding(seq_id, self.model_name, embedding, layer=layer_num)
                    elif isinstance(emb_item, (list, np.ndarray)):
                        embedding = np.array(emb_item)
                        if embedding.size > 0:
                            datastore.add_embedding(seq_id, self.model_name, embedding)

    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """
        Process sequences through prediction model.

        Performance: uses a single DuckDB anti-join for cache detection and a
        single JOIN query to merge predictions back — no per-row SQL calls.
        """
        start_count = len(df)

        # Ensure sequence_ids are present (normally added by _get_initial_data)
        if 'sequence_id' not in df.columns:
            df = df.copy()
            df['sequence_id'] = datastore.add_sequences_batch(df['sequence'].tolist())

        # --- Vectorized cache check: single anti-join, not N individual queries ---
        uncached_ids = datastore.get_uncached_sequence_ids(
            df['sequence_id'].tolist(), self.prediction_type, self.model_name
        )
        uncached_mask = df['sequence_id'].isin(uncached_ids)
        df_uncached = df[uncached_mask]
        cached_count = start_count - len(df_uncached)

        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)}")

        # Collect extra scalar fields returned alongside the main prediction value
        extra_fields: Dict[str, Dict[int, Any]] = {}

        if len(df_uncached) > 0:
            print(f"  Calling {self.model_name}.{self.action}...")

            # Reuse client across calls — DO NOT shut down in finally
            if self._api_client is None:
                self._api_client = BioLMApiClient(
                    self.model_name,
                    semaphore=self._semaphore,
                    retry_error_batches=True
                )
            api = self._api_client

            try:
                sequences = df_uncached['sequence'].tolist()
                seq_ids = df_uncached['sequence_id'].tolist()
                items = [{'sequence': seq} for seq in sequences]

                if self.action == 'encode':
                    results = await api.encode(items=items, params=self.params)
                else:
                    results = await api.predict(items=items, params=self.params)

                # Build batch insert list + collect extra fields in one pass
                batch_data = []
                for seq_id, result in zip(seq_ids, results):
                    if self.action == 'predict':
                        value = (
                            self._extract_prediction_value(result)
                            if isinstance(result, dict)
                            else (float(result) if result is not None else None)
                        )
                        batch_data.append({
                            'sequence_id': seq_id,
                            'prediction_type': self.prediction_type,
                            'model_name': self.model_name,
                            'value': value,
                            'metadata': {'params': self.params, 'result': result}
                            if isinstance(result, dict) else None,
                        })
                        # Collect additional scalar fields from the result dict
                        if isinstance(result, dict):
                            for key, val in result.items():
                                if isinstance(val, (int, float)) and key != 'value':
                                    extra_fields.setdefault(key, {})[seq_id] = val
                    elif self.action == 'encode':
                        self._store_embeddings(datastore, seq_id, result)

                # Single batch insert — one SQL call for all predictions
                if batch_data:
                    datastore.add_predictions_batch(batch_data)

            except Exception as e:
                if self.skip_on_error:
                    print(f"  Error during prediction (skipped): {e}")
                    failed_batch = [
                        {
                            'sequence_id': sid,
                            'prediction_type': self.prediction_type,
                            'model_name': self.model_name,
                            'value': None,
                            'metadata': {'status': 'failed', 'error': str(e), 'params': self.params},
                        }
                        for sid in df_uncached['sequence_id'].tolist()
                    ]
                    datastore.add_predictions_batch(failed_batch)
                else:
                    print(f"  Error during prediction: {e}")
                    raise
            # Note: no shutdown() — client is intentionally reused across batches

        # --- Vectorized result merge: single JOIN query, not N individual queries ---
        if self.action == 'predict':
            pred_df = datastore.get_predictions_bulk(
                df['sequence_id'].tolist(), self.prediction_type, self.model_name
            )
            if not pred_df.empty:
                df = df.merge(
                    pred_df[['sequence_id', 'value']].rename(columns={'value': self.prediction_type}),
                    on='sequence_id',
                    how='left',
                )
            else:
                df = df.copy()
                df[self.prediction_type] = None

            # Add any extra scalar fields collected during this run
            for field, id_to_val in extra_fields.items():
                if field not in df.columns:
                    df[field] = df['sequence_id'].map(id_to_val)

            df_out = df[df[self.prediction_type].notna()].copy()
            filtered_count = len(df) - len(df_out)

        elif self.action == 'encode':
            # Find sequences that received embeddings via single DuckDB query
            emb_ids = set(
                datastore.conn.execute(
                    "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ?",
                    [self.model_name],
                ).df()['sequence_id'].tolist()
            )
            df_out = df[df['sequence_id'].isin(emb_ids)].copy()
            filtered_count = len(df) - len(df_out)

        else:
            df_out = df.copy()
            filtered_count = 0

        return df_out, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_out),
            cached_count=cached_count,
            computed_count=len(df_uncached),
            filtered_count=filtered_count,
        )


class FilterStage(Stage):
    """
    Generic filtering stage.
    
    Args:
        name: Stage name
        filter_func: Filter function or BaseFilter instance
    """
    
    def __init__(self, name: str, filter_func: Union[BaseFilter, callable], **kwargs):
        super().__init__(name=name, **kwargs)
        self.filter_func = filter_func
        # Check if filter requires complete data (for streaming)
        self.requires_complete_data = getattr(
            filter_func, 'requires_complete_data', True  # Default: safe (batch)
        )
    
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Apply filter to DataFrame and return the filtered result."""
        start_count = len(df)

        print(f"  Applying filter: {self.filter_func}")
        df_filtered = self.filter_func(df)

        filtered_count = start_count - len(df_filtered)

        print(f"  Filtered out: {filtered_count}/{start_count}")
        print(f"  Remaining: {len(df_filtered)}")

        return df_filtered, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_filtered),
            filtered_count=filtered_count,
        )


class ClusteringStage(Stage):
    """
    Sequence clustering stage.
    
    Args:
        name: Stage name
        method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters (for kmeans/hierarchical)
        similarity_metric: How to measure similarity ('hamming', 'embedding')
        embedding_model: Model to use for embeddings (if similarity_metric='embedding')
    """
    
    def __init__(
        self,
        name: str,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        similarity_metric: str = 'hamming',
        embedding_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.embedding_model = embedding_model
        self.cluster_kwargs = kwargs
    
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Cluster sequences, add cluster columns, and return the enriched DataFrame."""
        from biolmai.pipeline.clustering import SequenceClusterer

        start_count = len(df)
        df = df.copy()

        print(f"  Clustering {start_count} sequences using {self.method}...")

        sequences = df['sequence'].tolist()

        # Load embeddings if needed — bulk fetch, not per-sequence
        embeddings = None
        if self.similarity_metric == 'embedding':
            if self.embedding_model is None:
                raise ValueError("embedding_model required when similarity_metric='embedding'")

            print(f"  Loading embeddings from {self.embedding_model}...")
            embeddings_list = []
            for seq in sequences:
                emb_list = datastore.get_embeddings_by_sequence(
                    seq, model_name=self.embedding_model, load_data=True
                )
                if not emb_list:
                    raise ValueError(f"No embedding found for sequence: {seq[:20]}...")
                # get_embeddings_by_sequence returns List[Dict]; access 'embedding' key
                emb_array = emb_list[0].get('embedding')
                if emb_array is None:
                    raise ValueError(
                        f"Embedding loaded but 'embedding' key missing for sequence: {seq[:20]}..."
                    )
                embeddings_list.append(emb_array)

            embeddings = np.stack(embeddings_list)

        # Perform clustering
        clusterer = SequenceClusterer(
            method=self.method,
            n_clusters=self.n_clusters,
            similarity_metric=self.similarity_metric,
            **self.cluster_kwargs
        )

        result = clusterer.cluster(sequences, embeddings)

        # Add cluster assignments to output DataFrame
        df['cluster_id'] = result.cluster_ids
        df['is_centroid'] = False
        df.loc[result.centroid_indices, 'is_centroid'] = True

        print(f"  Found {result.n_clusters} clusters")
        if result.silhouette_score is not None:
            print(f"  Silhouette score: {result.silhouette_score:.3f}")
        if result.davies_bouldin_score is not None:
            print(f"  Davies-Bouldin score: {result.davies_bouldin_score:.3f}")

        # Store clustering metadata via the dedicated method (uses pipeline_metadata table)
        meta = {
            'method': self.method,
            'n_clusters': result.n_clusters,
            'silhouette_score': result.silhouette_score,
            'davies_bouldin_score': result.davies_bouldin_score,
            'cluster_sizes': result.cluster_sizes,
        }
        datastore.set_pipeline_metadata(f"clustering_{self.name}", meta)

        return df, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=start_count,
            computed_count=start_count,
        )


class DataPipeline(BasePipeline):
    """
    Pipeline for processing existing sequences from files or lists.
    
    Load sequences from CSV/FASTA/lists and run predictions/filtering.
    
    Args:
        sequences: Input sequences (list of strings, DataFrame, or file path)
        datastore: DataStore instance or path
        run_id: Unique run ID
        output_dir: Output directory
        resume: Resume from previous run
        verbose: Enable verbose output
        diff_mode: If True, merge new sequences with existing cached results.
                  Only computes predictions for uncached sequences. Use get_merged_results()
                  or query_results() to efficiently access combined data without loading
                  millions of rows into memory (SQL-based queries).
    
    Example:
        >>> # Standard mode
        >>> pipeline = DataPipeline(sequences='sequences.csv')
        >>> pipeline.add_prediction('esmfold', prediction_type='structure')
        >>> pipeline.add_filter(ThresholdFilter('plddt', min_value=70))
        >>> results = pipeline.run()
        
        >>> # Diff mode - add new sequences to existing pipeline (SQL-based, efficient)
        >>> pipeline = DataPipeline(sequences='new_sequences.csv', diff_mode=True)
        >>> pipeline.add_prediction('esmfold', prediction_type='structure')
        >>> results = pipeline.run()
        >>> # Efficiently query specific data (doesn't load all millions of rows!)
        >>> high_quality = pipeline.query_results("s.length > 100 AND p.value > 70")
        >>> # Or get merged results with filters
        >>> merged = pipeline.get_merged_results(prediction_types=['plddt', 'tm'])
    """
    
    def __init__(
        self,
        sequences: Union[List[str], pd.DataFrame, str, Path] = None,
        diff_mode: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_sequences = sequences
        self.diff_mode = diff_mode
    
    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """Load sequences into DataFrame."""
        
        if self.input_sequences is None:
            raise ValueError("No sequences provided. Set 'sequences' parameter.")
        
        # Convert to DataFrame
        if isinstance(self.input_sequences, list):
            df = pd.DataFrame({'sequence': self.input_sequences})
        
        elif isinstance(self.input_sequences, pd.DataFrame):
            df = self.input_sequences.copy()
            if 'sequence' not in df.columns:
                raise ValueError("DataFrame must have 'sequence' column")
        
        elif isinstance(self.input_sequences, (str, Path)):
            path = Path(self.input_sequences)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if path.suffix == '.csv':
                df = pd.read_csv(path)
                if 'sequence' not in df.columns:
                    raise ValueError("CSV must have 'sequence' column")
            
            elif path.suffix in ['.fasta', '.fa', '.faa']:
                # Simple FASTA parser
                sequences = []
                current_seq = []
                
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if current_seq:
                                sequences.append(''.join(current_seq))
                                current_seq = []
                        else:
                            current_seq.append(line)
                    
                    if current_seq:
                        sequences.append(''.join(current_seq))
                
                df = pd.DataFrame({'sequence': sequences})
            
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        else:
            raise TypeError(f"Unsupported type for sequences: {type(self.input_sequences)}")
        
        # Deduplicate before inserting into datastore
        initial_count = len(df)
        df = df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
        deduplicated_count = initial_count - len(df)
        if deduplicated_count > 0 and self.verbose:
            print(f"Deduplicated {deduplicated_count} sequences ({len(df)} unique)")

        # Batch-add all sequences in one vectorized call instead of N individual calls
        df['sequence_id'] = self.datastore.add_sequences_batch(df['sequence'].tolist())

        # In diff mode, report how many sequences are new vs. already cached
        if self.diff_mode:
            existing_count = self.datastore.count_matching_sequences(df['sequence'].tolist())
            new_count = len(df) - existing_count
            if self.verbose:
                print(f"Diff mode: {existing_count} sequences already in datastore")
                print(f"Diff mode: {new_count} new sequences to process")
        
        return df
    
    def get_merged_results(
        self,
        prediction_types: Optional[List[str]] = None,
        sequence_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get results merged with existing cached data (for diff mode).
        
        This method is SQL-based and efficient - it doesn't load millions of rows.
        Instead, it queries only the data you need using DuckDB's columnar engine.
        
        Args:
            prediction_types: List of prediction types to include (None = all)
            sequence_filter: SQL WHERE clause to filter sequences (e.g., "length > 50")
        
        Returns:
            DataFrame with requested sequences and predictions
        
        Example:
            >>> # Get all results (efficient - DuckDB only loads what's needed)
            >>> df = pipeline.get_merged_results()
            
            >>> # Get only specific predictions (columnar - even faster!)
            >>> df = pipeline.get_merged_results(prediction_types=['tm', 'plddt'])
            
            >>> # Get sequences matching criteria (predicate pushdown!)
            >>> df = pipeline.get_merged_results(sequence_filter="length > 100")
        """
        if not self.diff_mode:
            # Standard mode: just return final data
            return self.get_final_data()
        
        # Use DuckDB's efficient query engine
        query = """
            SELECT 
                s.sequence_id,
                s.sequence,
                s.length
            FROM sequences s
        """
        
        if sequence_filter:
            query += f" WHERE {sequence_filter}"
        
        # Execute and load only requested data (columnar scan - fast!)
        df = self.datastore.query(query)
        
        if len(df) == 0:
            return df
        
        # Now load predictions for these sequences only
        if prediction_types:
            # Load specific prediction types
            for pred_type in prediction_types:
                self._add_predictions_to_df(df, pred_type)
        else:
            # Load all available prediction types for these sequences
            pred_types_df = self.datastore.query("""
                SELECT DISTINCT prediction_type 
                FROM predictions 
                WHERE sequence_id IN (SELECT sequence_id FROM df)
            """)
            
            pred_types = pred_types_df['prediction_type'].tolist()
            for pred_type in pred_types:
                self._add_predictions_to_df(df, pred_type)
        
        if self.verbose:
            print(f"\nDiff mode: Loaded {len(df)} sequences with predictions")
        
        return df
    
    def _add_predictions_to_df(
        self,
        df: pd.DataFrame,
        prediction_type: str
    ):
        """
        Efficiently add a prediction column to DataFrame using DuckDB SQL.
        Modifies df in place by leveraging DuckDB's join capabilities.
        """
        if len(df) == 0:
            return
        
        # Use DuckDB to efficiently join predictions
        # DuckDB can reference the DataFrame directly!
        result = self.datastore.query(f"""
            SELECT 
                df.sequence_id,
                p.value
            FROM df
            LEFT JOIN predictions p 
                ON df.sequence_id = p.sequence_id 
                AND p.prediction_type = '{prediction_type}'
        """)
        
        # Create dict for fast lookup
        pred_dict = dict(zip(result['sequence_id'], result['value']))
        
        # Add column to DataFrame
        df[prediction_type] = df['sequence_id'].map(pred_dict)
    
    def query_results(
        self,
        sql_where: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query results using SQL WHERE clause (for diff mode with large datasets).
        
        This leverages DuckDB's vectorized engine for maximum performance.
        
        Args:
            sql_where: SQL WHERE clause using table aliases:
                      - s.* for sequences table (e.g., "s.length > 100")
                      - Column names directly (no p. prefix needed)
            columns: Columns to include (None = all available)
        
        Returns:
            DataFrame with matching sequences (only loads what matches!)
        
        Example:
            >>> # Find long sequences (columnar scan - fast!)
            >>> df = pipeline.query_results("s.length > 200")
            
            >>> # Complex filter with predictions
            >>> df = pipeline.query_results(
            ...     "s.length > 100",
            ...     columns=['tm', 'plddt']
            ... )
        """
        # Build efficient DuckDB query
        query = f"""
            SELECT DISTINCT
                s.sequence_id,
                s.sequence,
                s.length
            FROM sequences s
            WHERE {sql_where}
        """
        
        # DuckDB executes with columnar scans and predicate pushdown
        df = self.datastore.query(query)
        
        # Add requested columns (predictions)
        if columns and len(df) > 0:
            for col in columns:
                self._add_predictions_to_df(df, col)
        
        return df
    
    def add_prediction(
        self,
        model_name: str,
        action: str = 'predict',
        prediction_type: Optional[str] = None,
        params: Optional[Dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add a prediction stage.
        
        Args:
            model_name: BioLM model name
            action: API action ('predict', 'encode', 'score')
            prediction_type: Type of prediction for caching
            params: Optional API parameters
            stage_name: Custom stage name (defaults to model_name)
            depends_on: List of stage names this depends on
        """
        stage_name = stage_name or f"{model_name}_{action}"
        
        stage = PredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            prediction_type=prediction_type,
            params=params,
            depends_on=depends_on or [],
            **kwargs
        )
        
        self.add_stage(stage)
        return self
    
    def add_predictions(
        self,
        models: List[Union[str, Dict]],
        action: str = 'predict',
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add multiple prediction stages at the same level (run in parallel).
        
        Args:
            models: List of model names or dicts with model configs
                   Example: ['esmfold', 'alphafold2']
                   Or: [{'model_name': 'esmfold', 'params': {...}}, ...]
            action: Default action if not specified in dict
            depends_on: List of stage names all these depend on
            **kwargs: Default kwargs for all stages
        
        Returns:
            self for chaining
        
        Example:
            >>> pipeline.add_predictions(['temberture', 'proteinmpnn', 'esm2'])
            >>> pipeline.add_predictions([
            ...     {'model_name': 'esmfold', 'prediction_type': 'structure'},
            ...     {'model_name': 'alphafold2', 'prediction_type': 'structure'}
            ... ])
        """
        for model in models:
            if isinstance(model, str):
                # Simple model name
                self.add_prediction(
                    model_name=model,
                    action=action,
                    depends_on=depends_on,
                    **kwargs
                )
            elif isinstance(model, dict):
                # Dict with config
                model_config = {**kwargs, **model}  # Model dict overrides defaults
                model_config['depends_on'] = model_config.get('depends_on', depends_on)
                
                model_name = model_config.pop('model_name')
                self.add_prediction(model_name=model_name, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")
        
        return self
    
    def add_filter(
        self,
        filter_func: Union[BaseFilter, callable],
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add a filter stage.
        
        Args:
            filter_func: Filter function or BaseFilter instance
            stage_name: Custom stage name
            depends_on: List of stage names this depends on
        """
        stage_name = stage_name or f"filter_{len(self.stages)}"
        
        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on or [],
            **kwargs
        )
        
        self.add_stage(stage)
        return self
    
    def add_clustering(
        self,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        similarity_metric: str = 'hamming',
        embedding_model: Optional[str] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Add a sequence clustering stage.
        
        Clusters sequences by similarity and adds cluster_id column to DataFrame.
        
        Args:
            method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (required for kmeans/hierarchical)
            similarity_metric: 'hamming' or 'embedding'
            embedding_model: Model name if using embedding similarity
            stage_name: Optional custom stage name
            depends_on: Optional list of stage names this stage depends on
            **kwargs: Additional arguments for clustering (eps, min_samples, etc.)
        
        Example:
            >>> # Cluster by sequence similarity
            >>> pipeline.add_clustering(method='kmeans', n_clusters=10)
            >>> 
            >>> # Cluster by embeddings
            >>> pipeline.add_prediction('esm2-650m', action='encode', stage_name='embed')
            >>> pipeline.add_clustering(
            ...     method='kmeans',
            ...     n_clusters=5,
            ...     similarity_metric='embedding',
            ...     embedding_model='esm2-650m',
            ...     depends_on=['embed']
            ... )
        """
        if stage_name is None:
            stage_name = f"cluster_{method}"
        
        stage = ClusteringStage(
            name=stage_name,
            method=method,
            n_clusters=n_clusters,
            similarity_metric=similarity_metric,
            embedding_model=embedding_model,
            depends_on=depends_on or [],
            **kwargs
        )
        
        self.add_stage(stage)
        return self


class SingleStepPipeline(DataPipeline):
    """
    Simplified pipeline for single-step predictions.
    
    Convenience class for running a single prediction model on sequences.
    
    Args:
        model_name: BioLM model name
        action: API action ('predict', 'encode', 'score')
        sequences: Input sequences
        params: Optional API parameters
        **kwargs: Additional arguments passed to DataPipeline
    
    Example:
        >>> pipeline = SingleStepPipeline(
        ...     model_name='esmfold',
        ...     sequences=['MKTAYIAKQRQ', 'MKLAVID']
        ... )
        >>> results = pipeline.run()
        >>> df = pipeline.get_final_data()
    """
    
    def __init__(
        self,
        model_name: str,
        action: str = 'predict',
        sequences: Union[List[str], pd.DataFrame, str, Path] = None,
        params: Optional[Dict] = None,
        prediction_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(sequences=sequences, **kwargs)
        
        # Automatically add single prediction stage
        self.add_prediction(
            model_name=model_name,
            action=action,
            prediction_type=prediction_type,
            params=params
        )


# Convenience aliases
def Predict(
    model_name: str,
    sequences: Union[List[str], pd.DataFrame, str, Path],
    params: Optional[Dict] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for single-step prediction.
    
    Args:
        model_name: BioLM model name
        sequences: Input sequences
        params: Optional API parameters
        **kwargs: Additional arguments
    
    Returns:
        DataFrame with predictions
    
    Example:
        >>> df = Predict('esmfold', sequences=['MKTAYIAKQRQ', 'MKLAVID'])
    """
    pipeline = SingleStepPipeline(
        model_name=model_name,
        action='predict',
        sequences=sequences,
        params=params,
        **kwargs
    )
    pipeline.run()
    return pipeline.get_final_data()


def Embed(
    model_name: str,
    sequences: Union[List[str], pd.DataFrame, str, Path],
    layer: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for generating embeddings.
    
    Args:
        model_name: BioLM model name (e.g., 'esm2')
        sequences: Input sequences
        layer: Optional layer number
        **kwargs: Additional arguments
    
    Returns:
        DataFrame with 'sequence', 'sequence_id', and 'embedding' columns
    
    Example:
        >>> df = Embed('esm2', sequences=['MKTAYIAKQRQ', 'MKLAVID'])
    """
    params = {'layer': layer} if layer is not None else None
    
    pipeline = SingleStepPipeline(
        model_name=model_name,
        action='encode',
        sequences=sequences,
        params=params,
        prediction_type='embedding',
        **kwargs
    )
    pipeline.run()
    df = pipeline.get_final_data()
    
    # Load embeddings into DataFrame for convenience
    embeddings_list = []
    for seq in df['sequence']:
        emb_list = pipeline.datastore.get_embeddings_by_sequence(seq, model_name=model_name, load_data=True)
        if emb_list:
            # get_embeddings_by_sequence returns List[Dict]; access 'embedding' key
            embeddings_list.append(emb_list[0].get('embedding'))
        else:
            embeddings_list.append(None)
    
    df['embedding'] = embeddings_list
    
    return df
