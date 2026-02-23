"""
Generative pipeline for sequence generation using language models.

Supports:
- Masked language models (ESM, ESM-1v) with remasking
- Inherently generative models (ProteinMPNN, ProGen2, etc.)
- Temperature and sampling parameter scanning
- Multi-model generation in parallel
"""

import json
import pandas as pd
import numpy as np
import asyncio
from typing import Union, List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import random

from biolmai.pipeline.base import BasePipeline, Stage, StageResult
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolmai.pipeline.mlm_remasking import MLMRemasker, RemaskingConfig
from biolmai.client import BioLMApiClient  # Use async client


@dataclass
class GenerationConfig:
    """
    Configuration for sequence generation.
    
    Args:
        model_name: BioLM model name
        num_sequences: Number of sequences to generate
        temperature: Temperature or list of temperatures to scan
        sampling_params: Additional sampling parameters
        generation_method: 'generate' or 'remask' (for MLMs)
        parent_sequence: Parent sequence (for remasking or conditioning)
        mask_positions: Positions to mask (for remasking), or 'auto' for automatic
        mask_fraction: Fraction of positions to mask (if mask_positions='auto')
        batch_size: Batch size for generation
    """
    model_name: str
    num_sequences: int = 100
    temperature: Union[float, List[float]] = 1.0
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    generation_method: str = 'generate'  # 'generate' or 'remask'
    parent_sequence: Optional[str] = None
    mask_positions: Union[str, List[int]] = 'auto'
    mask_fraction: float = 0.15
    batch_size: int = 32


class GenerationStage(Stage):
    """
    Stage for generating sequences using generative models.
    
    Args:
        name: Stage name
        configs: List of GenerationConfig objects (for multi-model generation)
        deduplicate: Whether to deduplicate generated sequences
    """
    
    def __init__(
        self,
        name: str = 'generation',
        configs: Optional[List[GenerationConfig]] = None,
        deduplicate: bool = True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.configs = configs or []
        self.deduplicate = deduplicate
    
    @staticmethod
    def _create_masked_sequence(
        sequence: str,
        mask_positions: Union[str, List[int]],
        mask_fraction: float = 0.15,
        mask_token: str = '<mask>'
    ) -> Tuple[str, List[int]]:
        """
        Create a masked version of a sequence.
        
        Args:
            sequence: Original sequence
            mask_positions: List of positions to mask, or 'auto' for random masking
            mask_fraction: Fraction of positions to mask (if mask_positions='auto')
            mask_token: Token to use for masking
        
        Returns:
            Tuple of (masked_sequence, masked_positions)
        """
        seq_list = list(sequence)
        
        if mask_positions == 'auto':
            # Randomly select positions to mask
            num_to_mask = max(1, int(len(sequence) * mask_fraction))
            positions = random.sample(range(len(sequence)), num_to_mask)
        else:
            positions = mask_positions
        
        # Apply masking
        for pos in positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = mask_token
        
        masked_seq = ''.join(seq_list)
        return masked_seq, sorted(positions)
    
    @staticmethod
    def _is_masked_lm(model_name: str) -> bool:
        """Check if a model is a masked language model."""
        mlm_models = ['esm', 'esm1v', 'esm2', 'esm1b']
        return any(mlm in model_name.lower() for mlm in mlm_models)
    
    async def _generate_with_config(
        self,
        config: GenerationConfig,
        datastore: DataStore
    ) -> List[Dict]:
        """
        Generate sequences with a single configuration.
        
        Returns:
            List of dicts with keys: sequence, model_name, temperature, sampling_params
        """
        results = []
        
        # Determine if we need temperature scanning
        temps = [config.temperature] if isinstance(config.temperature, (int, float)) else config.temperature
        
        for temp in temps:
            print(f"  {config.model_name} @ T={temp}: generating {config.num_sequences} sequences...")
            
            # Create async API client
            api = BioLMApiClient(config.model_name)
            
            try:
                if config.generation_method == 'remask' or (
                    config.generation_method == 'auto' and self._is_masked_lm(config.model_name)
                ):
                    # Remasking approach for MLMs
                    if config.parent_sequence is None:
                        raise ValueError("parent_sequence required for remasking")
                    
                    # Create RemaskingConfig from GenerationConfig
                    remask_config = RemaskingConfig(
                        model_name=config.model_name,  # Use model from GenerationConfig
                        mask_fraction=config.mask_fraction,
                        mask_positions=config.mask_positions,
                        num_iterations=config.sampling_params.get('num_iterations', 1),
                        temperature=temp,
                        top_k=config.sampling_params.get('top_k'),
                        top_p=config.sampling_params.get('top_p'),
                        conserved_positions=config.sampling_params.get('conserved_positions'),
                        mask_strategy=config.sampling_params.get('mask_strategy', 'random')
                    )
                    
                    # Create remasker
                    remasker = MLMRemasker(
                        remask_config,
                        api_client=api,
                        model_name=config.model_name
                    )
                    
                    # Generate variants (now async)
                    variants = await remasker.generate_variants(
                        config.parent_sequence,
                        num_variants=config.num_sequences,
                        deduplicate=True
                    )
                    
                    # Store results
                    for seq, metadata in variants:
                        results.append({
                            'sequence': seq,
                            'model_name': config.model_name,
                            'temperature': temp,
                            'sampling_params': config.sampling_params,
                            'generation_method': 'remask',
                            'parent_sequence': config.parent_sequence,
                            'num_mutations': metadata.get('num_mutations'),
                            'mutation_rate': metadata.get('mutation_rate')
                        })
                
                else:
                    # Direct generation (for inherently generative models)
                    params = {
                        **config.sampling_params,
                        'temperature': temp,
                        'num_sequences': config.num_sequences
                    }
                    
                    # Prepare items
                    if config.parent_sequence:
                        items = [{'sequence': config.parent_sequence}]
                    else:
                        items = [{}]  # Unconditional generation
                    
                    # Generate (now async)
                    result = await api.generate(items=items, params=params)
                    
                    # Extract sequences
                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict):
                                seq = item.get('sequence', str(item))
                            else:
                                seq = str(item)
                            
                            results.append({
                                'sequence': seq,
                                'model_name': config.model_name,
                                'temperature': temp,
                                'sampling_params': config.sampling_params,
                                'generation_method': 'generate',
                                'parent_sequence': config.parent_sequence
                            })
            
            finally:
                await api.shutdown()
        
        return results
    
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Generate sequences using configured models."""

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._generate_with_config(self.configs[0], datastore)
        else:
            tasks = [self._generate_with_config(cfg, datastore) for cfg in self.configs]
            results_list = await asyncio.gather(*tasks)
            results = [item for sublist in results_list for item in sublist]

        df_generated = pd.DataFrame(results) if results else pd.DataFrame(
            columns=['sequence', 'model_name', 'temperature', 'sampling_params',
                     'generation_method', 'parent_sequence']
        )
        initial_count = len(df_generated)

        if self.deduplicate and initial_count > 0:
            df_generated = df_generated.drop_duplicates(subset=['sequence']).reset_index(drop=True)
            deduplicated_count = initial_count - len(df_generated)
            if deduplicated_count > 0:
                print(f"  Deduplicated: {deduplicated_count} sequences ({len(df_generated)} unique)")

        if len(df_generated) > 0:
            print(f"  Adding {len(df_generated)} sequences to datastore...")

            # Batch-insert all sequences in one vectorized call
            seq_ids = datastore.add_sequences_batch(df_generated['sequence'].tolist())
            df_generated['sequence_id'] = seq_ids

            # Store generation metadata per sequence
            for row_idx, (seq_id, row) in enumerate(
                zip(seq_ids, df_generated.itertuples(index=False))
            ):
                sampling_params = (getattr(row, 'sampling_params', None) or {})
                datastore.add_generation_metadata(
                    seq_id,
                    model_name=row.model_name,
                    temperature=getattr(row, 'temperature', None),
                    top_k=sampling_params.get('top_k'),
                    top_p=sampling_params.get('top_p'),
                    num_return_sequences=sampling_params.get('num_return_sequences'),
                    do_sample=sampling_params.get('do_sample'),
                    repetition_penalty=sampling_params.get('repetition_penalty'),
                    max_length=sampling_params.get('max_length'),
                )

        return df_generated, StageResult(
            stage_name=self.name,
            input_count=0,
            output_count=len(df_generated),
            computed_count=initial_count,
            metadata={'deduplicated': initial_count - len(df_generated) if self.deduplicate else 0},
        )


class GenerativePipeline(BasePipeline):
    """
    Pipeline for generating sequences and running predictions.
    
    Supports:
    - Multiple generative models in parallel
    - Temperature scanning
    - Masked language model remasking
    - Downstream predictions and filtering
    
    Args:
        generation_configs: List of GenerationConfig objects
        deduplicate: Whether to deduplicate generated sequences
        datastore: DataStore instance or path
        run_id: Unique run ID
        output_dir: Output directory
        resume: Resume from previous run
        verbose: Enable verbose output
    
    Example:
        >>> # Generate with MPNN at multiple temperatures
        >>> config1 = GenerationConfig(
        ...     model_name='proteinmpnn',
        ...     num_sequences=1000,
        ...     temperature=[0.5, 1.0, 1.5],
        ...     parent_sequence='MKTAYIAKQRQ'
        ... )
        >>> 
        >>> # Also generate with ESM using remasking
        >>> config2 = GenerationConfig(
        ...     model_name='esm2',
        ...     num_sequences=500,
        ...     generation_method='remask',
        ...     parent_sequence='MKTAYIAKQRQ',
        ...     mask_fraction=0.15
        ... )
        >>> 
        >>> pipeline = GenerativePipeline(
        ...     generation_configs=[config1, config2]
        ... )
        >>> pipeline.add_filter(ThresholdFilter('length', min_value=50))
        >>> pipeline.add_prediction('esmfold', prediction_type='structure')
        >>> results = pipeline.run()
    """
    
    def __init__(
        self,
        generation_configs: Optional[List[GenerationConfig]] = None,
        deduplicate: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.generation_configs = generation_configs or []
        self.deduplicate = deduplicate
        
        # Add generation stage automatically
        if self.generation_configs:
            gen_stage = GenerationStage(
                name='generation',
                configs=self.generation_configs,
                deduplicate=self.deduplicate
            )
            self.stages.insert(0, gen_stage)
    
    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """
        For generative pipeline, initial data comes from generation stage.
        
        This method is called after generation stage completes.
        """
        # Generation stage will populate _stage_data['generation']
        if 'generation' in self._stage_data:
            return self._stage_data['generation']
        
        # If no generation configs, return empty DataFrame
        return pd.DataFrame(columns=['sequence', 'sequence_id'])
    
    def add_generation_config(self, config: GenerationConfig):
        """Add a generation configuration."""
        self.generation_configs.append(config)
        
        # Update generation stage if it exists
        if self.stages and isinstance(self.stages[0], GenerationStage):
            self.stages[0].configs = self.generation_configs
        else:
            # Create generation stage
            gen_stage = GenerationStage(
                name='generation',
                configs=self.generation_configs,
                deduplicate=self.deduplicate
            )
            self.stages.insert(0, gen_stage)
        
        return self
    
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
        """Add a prediction stage (same as DataPipeline)."""
        from biolmai.pipeline.data import PredictionStage
        
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
            action: Default action if not specified in dict
            depends_on: List of stage names all these depend on
            **kwargs: Default kwargs for all stages
        
        Returns:
            self for chaining
        
        Example:
            >>> pipeline.add_predictions(['temberture', 'proteinmpnn', 'esm2'])
        """
        from biolmai.pipeline.data import PredictionStage
        
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
                model_config = {**kwargs, **model}
                model_config['depends_on'] = model_config.get('depends_on', depends_on)
                
                model_name = model_config.pop('model_name')
                self.add_prediction(model_name=model_name, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")
        
        return self
    
    def add_filter(
        self,
        filter_func,
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ):
        """Add a filter stage (same as DataPipeline)."""
        from biolmai.pipeline.data import FilterStage
        
        stage_name = stage_name or f"filter_{len(self.stages)}"
        
        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on or [],
            **kwargs
        )
        
        self.add_stage(stage)
        return self
    
    async def run_async(self, **kwargs) -> Dict[str, StageResult]:
        """
        Run the generative pipeline.
        
        This override handles the special case of generation stage.
        """
        # Run generation stage first if it exists
        if self.stages and isinstance(self.stages[0], GenerationStage):
            gen_stage = self.stages[0]
            
            print(f"\n{'='*60}")
            print(f"Stage: {gen_stage.name} (Generation)")
            print(f"Configs: {len(gen_stage.configs)}")
            print(f"{'='*60}")
            
            # Execute generation — returns (df_generated, StageResult)
            df_generated, result = await gen_stage.process(pd.DataFrame(), self.datastore)
            self.stage_results[gen_stage.name] = result

            # If the generated DataFrame is empty or missing sequence_id, fall back to
            # a DuckDB export that includes generation metadata columns
            if df_generated.empty or 'sequence_id' not in df_generated.columns:
                df_generated = self.datastore.export_to_dataframe(
                    include_sequences=True,
                    include_predictions=False,
                    include_generation_metadata=True,
                )

            self._stage_data[gen_stage.name] = df_generated
            
            if self.verbose:
                print(f"\n{result}")
                print(f"{'='*60}")
            
            # Remove generation stage from list (so base class doesn't run it again)
            remaining_stages = self.stages[1:]
            self.stages = remaining_stages
        
        # Run remaining stages using base class
        return await super().run_async(**kwargs)


# Convenience function for quick generation
def Generate(
    model_name: str,
    num_sequences: int = 100,
    temperature: Union[float, List[float]] = 1.0,
    parent_sequence: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for quick sequence generation.
    
    Args:
        model_name: BioLM model name
        num_sequences: Number of sequences to generate
        temperature: Temperature or list of temperatures
        parent_sequence: Optional parent sequence
        **kwargs: Additional arguments
    
    Returns:
        DataFrame with generated sequences
    
    Example:
        >>> df = Generate('proteinmpnn', num_sequences=100, temperature=[0.5, 1.0])
    """
    config = GenerationConfig(
        model_name=model_name,
        num_sequences=num_sequences,
        temperature=temperature,
        parent_sequence=parent_sequence
    )
    
    pipeline = GenerativePipeline(generation_configs=[config], **kwargs)
    pipeline.run()
    return pipeline.get_final_data()
