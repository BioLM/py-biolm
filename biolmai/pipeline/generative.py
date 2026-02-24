"""
Generative pipeline for sequence generation using language models.

Supports:
- Masked language models (ESM, ESM-1v) with remasking
- Inherently generative models (ProteinMPNN, ProGen2, etc.)
- Temperature and sampling parameter scanning
- Multi-model generation in parallel
"""

import json
import warnings
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
class DirectGenerationConfig:
    """
    Configuration for structure-conditioned sequence generation.

    Use with models such as ProteinMPNN, AntiFold, HyperMPNN, LigandMPNN.

    Args:
        model_name: BioLM model name (e.g. 'proteinmpnn', 'antifold').
        structure_path: Path to a PDB or CIF file (first run / static structure).
        structure_column: Column in the upstream DataFrame that holds structure
            strings (for chained pipelines where structure was predicted upstream).
        num_sequences: Number of sequences to generate per structure.
        temperature: Sampling temperature.
        top_k: Top-k sampling (optional).
        chain_id: Chain to redesign (AntiFold / HyperMPNN).
        fixed_positions: 0-indexed residue positions to keep fixed.
    """
    model_name: str
    structure_path: Optional[str] = None
    structure_column: Optional[str] = None
    num_sequences: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    chain_id: Optional[str] = None
    fixed_positions: Optional[List[int]] = None


@dataclass
class GenerationConfig:
    """
    Configuration for sequence generation.

    .. deprecated::
        Use :class:`RemaskingConfig` for MLM-based generation or
        :class:`DirectGenerationConfig` for structure-conditioned models.

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

    def __post_init__(self):
        warnings.warn(
            "GenerationConfig is deprecated. Use RemaskingConfig for MLM remasking "
            "or DirectGenerationConfig for structure-conditioned generation.",
            DeprecationWarning,
            stacklevel=2,
        )


class GenerationStage(Stage):
    """
    Stage for generating sequences using generative models.

    Accepts either the new typed configs (RemaskingConfig / DirectGenerationConfig)
    or the legacy GenerationConfig list.

    Args:
        name: Stage name.
        config: Single RemaskingConfig or DirectGenerationConfig (new API).
        configs: List of config objects — GenerationConfig, RemaskingConfig, or
                 DirectGenerationConfig (old / multi-model API).
        deduplicate: Whether to deduplicate generated sequences.
    """

    def __init__(
        self,
        name: str = 'generation',
        config: Optional[Union[RemaskingConfig, DirectGenerationConfig]] = None,
        configs: Optional[List[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]] = None,
        deduplicate: bool = True,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if config is not None:
            self.configs = [config]
        elif configs is not None:
            self.configs = configs
        else:
            self.configs = []
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
    
    async def _run_remasking(
        self,
        config: RemaskingConfig,
        parent_sequence: str,
        num_variants: int,
    ) -> List[Dict]:
        """Run MLM remasking generation using RemaskingConfig directly."""
        print(f"  {config.model_name} (remasking): generating {num_variants} variants...")
        api = BioLMApiClient(config.model_name)
        try:
            remasker = MLMRemasker(config, api_client=api, model_name=config.model_name)
            variants = await remasker.generate_variants(
                parent_sequence,
                num_variants=num_variants,
                deduplicate=True,
            )
            return [
                {
                    'sequence': seq,
                    'model_name': config.model_name,
                    'temperature': config.temperature,
                    'sampling_params': {},
                    'generation_method': 'remask',
                    'parent_sequence': parent_sequence,
                    'num_mutations': meta.get('num_mutations'),
                    'mutation_rate': meta.get('mutation_rate'),
                }
                for seq, meta in variants
            ]
        finally:
            await api.shutdown()

    async def _run_direct_generation(
        self,
        config: DirectGenerationConfig,
        datastore: DataStore,
        df_input: pd.DataFrame,
    ) -> List[Dict]:
        """Run structure-conditioned generation using DirectGenerationConfig."""
        # Resolve structure string(s)
        if config.structure_column and config.structure_column in df_input.columns:
            structure_strings = df_input[config.structure_column].dropna().tolist()
        elif config.structure_path:
            from biolmai.pipeline.utils import load_structure_string, cif_to_pdb
            import tempfile
            fmt, structure_str = load_structure_string(config.structure_path)
            if fmt == 'cif':
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as f:
                    pdb_path = f.name
                cif_to_pdb(config.structure_path, pdb_path)
                _, structure_str = load_structure_string(pdb_path)
            structure_strings = [structure_str]
        else:
            raise ValueError(
                "DirectGenerationConfig requires either structure_path or structure_column"
            )

        params: Dict[str, Any] = {
            'num_sequences': config.num_sequences,
            'temperature': config.temperature,
        }
        if config.top_k is not None:
            params['top_k'] = config.top_k
        if config.chain_id:
            params['chain_id'] = config.chain_id
        if config.fixed_positions:
            params['fixed_positions'] = config.fixed_positions

        results = []
        api = BioLMApiClient(config.model_name)
        try:
            for structure_str in structure_strings:
                print(f"  {config.model_name} (direct): generating {config.num_sequences} sequences...")
                items = [{'structure': structure_str}]
                raw = await api.generate(items=items, params=params)
                if isinstance(raw, list):
                    for item in raw:
                        seq = item.get('sequence', str(item)) if isinstance(item, dict) else str(item)
                        results.append({
                            'sequence': seq,
                            'model_name': config.model_name,
                            'temperature': config.temperature,
                            'sampling_params': params,
                            'generation_method': 'direct',
                            'parent_sequence': None,
                        })
        finally:
            await api.shutdown()

        return results

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
    
    async def _dispatch_config(
        self,
        cfg: Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig],
        datastore: DataStore,
        df_input: pd.DataFrame,
    ) -> List[Dict]:
        """Dispatch a single config to the appropriate generation method."""
        if isinstance(cfg, RemaskingConfig):
            parent = getattr(cfg, 'parent_sequence', None)
            num = getattr(cfg, 'num_variants', 100)
            if parent is None:
                raise ValueError("RemaskingConfig requires a parent_sequence attribute")
            return await self._run_remasking(cfg, parent_sequence=parent, num_variants=num)
        elif isinstance(cfg, DirectGenerationConfig):
            return await self._run_direct_generation(cfg, datastore, df_input)
        elif isinstance(cfg, GenerationConfig):
            return await self._generate_with_config(cfg, datastore)
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")

    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Generate sequences using configured models."""

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._dispatch_config(self.configs[0], datastore, df)
        else:
            tasks = [self._dispatch_config(cfg, datastore, df) for cfg in self.configs]
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
        generation_configs: Optional[List[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]] = None,
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.datastore:
            self.datastore.close()
        return False
    
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
        Idempotent: self.stages is always restored after execution.
        """
        import copy as _copy

        # Save original stages so the pipeline is idempotent (can be called again)
        original_stages = self.stages

        try:
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
                self._stage_data[gen_stage.name] = df_generated

                if self.verbose:
                    print(f"\n{result}")
                    print(f"{'='*60}")

                # Build patched stage list without mutating stage objects or self.stages.
                # Each remaining stage gets a shallow copy with depends_on stripped of
                # the generation stage name, so base-class dependency resolution works.
                gen_name = gen_stage.name
                remaining_stages = []
                for s in self.stages[1:]:
                    s_copy = _copy.copy(s)
                    s_copy.depends_on = [d for d in s.depends_on if d != gen_name]
                    remaining_stages.append(s_copy)
                self.stages = remaining_stages

            # Run remaining stages using base class
            return await super().run_async(**kwargs)
        finally:
            # Always restore original stages so the pipeline remains idempotent
            self.stages = original_stages


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
