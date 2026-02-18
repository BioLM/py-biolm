"""
Data-driven pipeline implementations.

DataPipeline: Load sequences from files/lists and run predictions
SingleStepPipeline: Simplified single-step prediction pipeline
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

from biolmai.pipeline.base import BasePipeline, Stage, StageResult
from biolmai.pipeline.datastore import DataStore
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
        # Reuse API client across calls for connection pooling
        self._api_client = None
    
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> StageResult:
        """Process sequences through prediction model."""
        
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
        print(f"  To compute: {len(df_uncached)}")
        
        if len(df_uncached) > 0:
            # Process uncached sequences
            print(f"  Calling {self.model_name}.{self.action}...")
            
            # Create or reuse async BioLM API client with shared semaphore
            if self._api_client is None:
                self._api_client = BioLMApiClient(
                    self.model_name,
                    semaphore=self._semaphore,  # Share stage's semaphore for rate limiting
                    retry_error_batches=True  # Auto-retry failed batches individually
                )
            api = self._api_client
            
            try:
                # Prepare items for API
                sequences = df_uncached['sequence'].tolist()
                items = [{'sequence': seq} for seq in sequences]
                
                # Call appropriate async API method based on action
                if self.action == 'encode':
                    results = await api.encode(items=items, params=self.params)
                else:  # predict
                    results = await api.predict(items=items, params=self.params)
                
                # Store results in datastore
                for seq, result in zip(sequences, results):
                    seq_id = datastore.add_sequence(seq)
                    
                    if self.action == 'predict':
                        # Extract value from result based on model
                        if isinstance(result, dict):
                            # Try model-specific fields first
                            if 'melting_temperature' in result:
                                value = result['melting_temperature']
                            elif 'solubility_score' in result:
                                value = result['solubility_score']
                            elif 'prediction' in result:
                                value = result['prediction']
                            elif 'score' in result:
                                value = result['score']
                            elif 'value' in result:
                                value = result['value']
                            else:
                                # Use first numeric value found
                                value = next((v for v in result.values() if isinstance(v, (int, float))), None)
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
                    
                    elif self.action == 'encode':
                        # Store embedding(s)
                        if isinstance(result, dict):
                            # Handle different embedding formats
                            
                            if 'embedding' in result:
                                # Direct embedding field - single embedding
                                embedding = np.array(result['embedding'])
                                datastore.add_embedding(seq_id, self.model_name, embedding)
                                
                            elif 'embeddings' in result:
                                # Nested embeddings (e.g., ESM2 returns list of layer embeddings)
                                embs = result['embeddings']
                                
                                if isinstance(embs, list):
                                    # Store each layer separately
                                    for emb_item in embs:
                                        if isinstance(emb_item, dict):
                                            layer_num = emb_item.get('layer', None)
                                            emb_data = emb_item.get('embedding', None)
                                            
                                            if emb_data is not None:
                                                embedding = np.array(emb_data)
                                                if len(embedding) > 0:
                                                    datastore.add_embedding(
                                                        seq_id, 
                                                        self.model_name, 
                                                        embedding,
                                                        layer=layer_num
                                                    )
                                        else:
                                            # Plain array
                                            embedding = np.array(emb_item)
                                            if len(embedding) > 0:
                                                datastore.add_embedding(seq_id, self.model_name, embedding)
                                
                                elif isinstance(embs, (np.ndarray, list)):
                                    # Direct array
                                    embedding = np.array(embs)
                                    if len(embedding) > 0:
                                        datastore.add_embedding(seq_id, self.model_name, embedding)
                
            finally:
                api.shutdown()
        
        # Merge predictions back into DataFrame (for predict action)
        if self.action == 'predict':
            def get_prediction(row):
                preds = datastore.get_predictions_by_sequence(row['sequence'])
                if not preds.empty:
                    matching = preds[
                        (preds['prediction_type'] == self.prediction_type) &
                        (preds['model_name'] == self.model_name)
                    ]
                    if not matching.empty:
                        return matching.iloc[0]['value']
                return None
            
            df[self.prediction_type] = df.apply(get_prediction, axis=1)
            
            # Also add individual result fields to DataFrame
            for idx, row in df.iterrows():
                preds = datastore.get_predictions_by_sequence(row['sequence'])
                if not preds.empty:
                    matching = preds[
                        (preds['prediction_type'] == self.prediction_type) &
                        (preds['model_name'] == self.model_name)
                    ]
                    if not matching.empty:
                        pred_row = matching.iloc[0]
                        if 'metadata' in pred_row and pred_row['metadata']:
                            try:
                                import json
                                metadata = json.loads(pred_row['metadata']) if isinstance(pred_row['metadata'], str) else pred_row['metadata']
                                if 'result' in metadata and isinstance(metadata['result'], dict):
                                    # Add all fields from result to dataframe
                                    for key, val in metadata['result'].items():
                                        if isinstance(val, (int, float, str, bool)):
                                            if key not in df.columns:
                                                df[key] = None
                                            df.at[idx, key] = val
                            except:
                                pass
            
            # Filter out sequences with no prediction
            df_with_pred = df[df[self.prediction_type].notna()].copy()
            filtered_count = len(df) - len(df_with_pred)
        
        elif self.action == 'encode':
            # For embeddings, check if embedding was created and filter
            def has_embedding(row):
                embs = datastore.get_embeddings_by_sequence(row['sequence'])
                if embs:  # embs is a list
                    return any(e['model_name'] == self.model_name for e in embs)
                return False
            
            # Filter to only sequences with embeddings
            mask = df.apply(has_embedding, axis=1)
            initial_len = len(df)
            df.drop(df[~mask].index, inplace=True)
            filtered_count = initial_len - len(df)
            df_with_pred = df  # For consistency with other branches
        
        else:
            # For other actions (structure), don't filter
            df_with_pred = df.copy()
            filtered_count = 0
        
        return StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_with_pred),
            cached_count=cached_count,
            computed_count=len(df_uncached),
            filtered_count=filtered_count
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
    
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> StageResult:
        """Apply filter to DataFrame."""
        
        start_count = len(df)
        
        print(f"  Applying filter: {self.filter_func}")
        df_filtered = self.filter_func(df)
        
        filtered_count = start_count - len(df_filtered)
        
        print(f"  Filtered out: {filtered_count}/{start_count}")
        print(f"  Remaining: {len(df_filtered)}")
        
        return StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_filtered),
            filtered_count=filtered_count
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
    ) -> StageResult:
        """Cluster sequences and add cluster assignments to DataFrame."""
        from biolmai.pipeline.clustering import SequenceClusterer
        
        start_count = len(df)
        
        print(f"  Clustering {start_count} sequences using {self.method}...")
        
        sequences = df['sequence'].tolist()
        
        # Get embeddings if needed
        embeddings = None
        if self.similarity_metric == 'embedding':
            if self.embedding_model is None:
                raise ValueError("embedding_model required when similarity_metric='embedding'")
            
            print(f"  Loading embeddings from {self.embedding_model}...")
            embeddings_list = []
            for seq in sequences:
                emb_list = datastore.get_embeddings_by_sequence(
                    seq,
                    model_name=self.embedding_model,
                    load_data=True
                )
                if emb_list:
                    _, embedding = emb_list[0]
                    embeddings_list.append(embedding)
                else:
                    raise ValueError(f"No embedding found for sequence: {seq[:20]}...")
            
            embeddings = np.stack(embeddings_list)
        
        # Perform clustering
        clusterer = SequenceClusterer(
            method=self.method,
            n_clusters=self.n_clusters,
            similarity_metric=self.similarity_metric,
            **self.cluster_kwargs
        )
        
        result = clusterer.cluster(sequences, embeddings)
        
        # Add cluster assignments to DataFrame
        df['cluster_id'] = result.cluster_ids
        df['is_centroid'] = False
        df.loc[result.centroid_indices, 'is_centroid'] = True
        
        print(f"  Found {result.n_clusters} clusters")
        if result.silhouette_score is not None:
            print(f"  Silhouette score: {result.silhouette_score:.3f}")
        if result.davies_bouldin_score is not None:
            print(f"  Davies-Bouldin score: {result.davies_bouldin_score:.3f}")
        
        # Store clustering metadata
        metadata = {
            'method': self.method,
            'n_clusters': result.n_clusters,
            'silhouette_score': result.silhouette_score,
            'davies_bouldin_score': result.davies_bouldin_score,
            'cluster_sizes': result.cluster_sizes
        }
        
        # Store in datastore (as JSON metadata)
        import json
        datastore.conn.execute(
            "INSERT OR REPLACE INTO pipeline_metadata (key, value) VALUES (?, ?)",
            (f"clustering_{self.name}", json.dumps(metadata))
        )
        datastore.conn.commit()
        
        return StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=start_count,
            computed_count=start_count
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
    
    Example:
        >>> pipeline = DataPipeline(sequences='sequences.csv')
        >>> pipeline.add_prediction('esmfold', prediction_type='structure')
        >>> pipeline.add_filter(ThresholdFilter('plddt', min_value=70))
        >>> pipeline.add_prediction('temberture', prediction_type='tm')
        >>> results = pipeline.run()
    """
    
    def __init__(
        self,
        sequences: Union[List[str], pd.DataFrame, str, Path] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_sequences = sequences
    
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
        
        # Add sequence IDs
        df['sequence_id'] = df['sequence'].apply(
            lambda seq: self.datastore.add_sequence(seq)
        )
        
        # Deduplicate
        initial_count = len(df)
        df = df.drop_duplicates(subset=['sequence']).reset_index(drop=True)
        deduplicated_count = initial_count - len(df)
        
        if deduplicated_count > 0 and self.verbose:
            print(f"Deduplicated {deduplicated_count} sequences ({len(df)} unique)")
        
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
            _, embedding = emb_list[0]  # Use first (or only) embedding
            embeddings_list.append(embedding)
        else:
            embeddings_list.append(None)
    
    df['embedding'] = embeddings_list
    
    return df
