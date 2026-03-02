"""
Generative pipeline for sequence generation using language models.

Supports:
- Masked language models (ESM, ESM-1v) with remasking
- Inherently generative models (ProteinMPNN, ProGen2, etc.)
- Temperature and sampling parameter scanning
- Multi-model generation in parallel
"""

import asyncio
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from biolmai.client import BioLMApiClient  # Use async client
from biolmai.pipeline.base import BasePipeline, Stage, StageResult
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolmai.pipeline.mlm_remasking import MLMRemasker, RemaskingConfig


@dataclass
class DirectGenerationConfig:
    """
    Configuration for structure- or sequence-conditioned generation.

    Use with models such as ProteinMPNN, AntiFold, HyperMPNN, LigandMPNN, DSM.

    The caller is responsible for providing the correct ``item_field`` and
    ``params`` for the target model — these vary per model and are documented in
    the BioLM API schema (``/schema/<model>/generate/``).  Common values:

    +------------------------+-------------+-------------------------------------------+
    | Model                  | item_field  | Key params                                |
    +========================+=============+===========================================+
    | protein-mpnn, hyper-   | ``'pdb'``   | ``batch_size``, ``temperature``           |
    | mpnn, soluble-mpnn,    |             |                                           |
    | ligand-mpnn,           |             |                                           |
    | global-label-membrane- |             |                                           |
    | mpnn                   |             |                                           |
    +------------------------+-------------+-------------------------------------------+
    | antifold               | ``'pdb'``   | ``heavy_chain``, ``light_chain``,         |
    |                        |             | ``num_seq_per_target``, ``sampling_temp`` |
    +------------------------+-------------+-------------------------------------------+
    | dsm-150m-base, dsm-    | ``'sequence'``| ``num_sequences``, ``temperature``      |
    | 650m-base              |             |                                           |
    +------------------------+-------------+-------------------------------------------+

    Args:
        model_name: BioLM model slug (e.g. ``'protein-mpnn'``, ``'antifold'``,
            ``'dsm-150m-base'``).
        structure_path: Path to a PDB or CIF file.
        structure_column: DataFrame column holding PDB strings (for chained
            pipelines where structure was predicted upstream).
        sequence: Parent sequence string for sequence-conditioned models (DSM).
        item_field: The item dict key expected by the model API — ``'pdb'`` for
            structure-conditioned models, ``'sequence'`` for DSM.  Defaults to
            ``'pdb'``.
        params: **Model-specific params dict (required for non-trivial calls).**
            Keys must exactly match the model's API param names.  When empty the
            stage sends ``{'num_sequences': num_sequences, 'temperature': temperature}``
            as a simple fallback — useful only for models that accept these exact
            param names.
        num_sequences: Fallback when ``params`` is empty (default 100).
        temperature: Fallback when ``params`` is empty (default 1.0).
    """

    model_name: str
    structure_path: Optional[str] = None
    structure_column: Optional[str] = None
    sequence: Optional[str] = None
    item_field: str = "pdb"
    params: Dict[str, Any] = field(default_factory=dict)
    # Simple fallbacks used only when params is empty
    num_sequences: int = 100
    temperature: float = 1.0


@dataclass
class FoldingEntity:
    """
    A molecular entity for co-folding models (Boltz2, Chai-1).

    Used with :meth:`DataPipeline.add_cofolding_prediction` and
    :meth:`GenerativePipeline.add_cofolding_prediction` to inject static
    entities — ligands, cofactors, DNA/RNA strands, or additional protein
    chains — alongside the pipeline's primary sequences.

    Args:
        id: Chain identifier (Boltz uses single letter(s); Chai-1 uses a name).
        entity_type: Molecule type — ``'protein'``, ``'dna'``, ``'rna'``, or
            ``'ligand'``.
        sequence: Amino-acid / nucleotide sequence for protein / DNA / RNA entities.
        smiles: SMILES string for small-molecule ligands.
        ccd: CCD code for ligands defined in the Chemical Component Dictionary
            (e.g. ``'ATP'``, ``'HEM'``).
    """

    id: str
    entity_type: str  # 'protein' | 'dna' | 'rna' | 'ligand'
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd: Optional[str] = None


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
    generation_method: str = "generate"  # 'generate' or 'remask'
    parent_sequence: Optional[str] = None
    mask_positions: Union[str, List[int]] = "auto"
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

    For ``DirectGenerationConfig`` the caller supplies the correct ``item_field``
    (e.g. ``'pdb'`` or ``'sequence'``) and ``params`` dict with model-specific param
    names — no auto-detection is performed.

    The stage handles the three main response formats returned by BioLM generation
    models:

    * **Flat list** (MPNN): ``[{sequence, pdb, ...}, ...]``
    * **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
    * **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``

    Args:
        name: Stage name.
        config: Single RemaskingConfig or DirectGenerationConfig (new API).
        configs: List of config objects — GenerationConfig, RemaskingConfig, or
                 DirectGenerationConfig (old / multi-model API).
        deduplicate: Whether to deduplicate generated sequences.
    """

    def __init__(
        self,
        name: str = "generation",
        config: Optional[Union[RemaskingConfig, DirectGenerationConfig]] = None,
        configs: Optional[
            List[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]
        ] = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if config is not None:
            self.configs = [config]
        elif configs is not None:
            self.configs = configs
        else:
            self.configs = []
        self.deduplicate = deduplicate

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sequences(raw: Any) -> List[Dict]:
        """
        Extract sequence dicts from any BioLM generation response format.

        Handles:
        - **Flat list** (MPNN): ``[{sequence, pdb, ...}, ...]``
        - **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
        - **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``
        """
        if not raw:
            return []
        sequences: List[Dict] = []
        for item in raw:
            if isinstance(item, list):
                # DSM-style: inner list = num_sequences per input item
                for sub in item:
                    if isinstance(sub, dict):
                        seq = sub.get("sequence", "")
                        if seq:
                            sequences.append(
                                {
                                    "sequence": seq,
                                    "log_prob": sub.get("log_prob"),
                                    "perplexity": sub.get("perplexity"),
                                }
                            )
            elif isinstance(item, dict):
                if "sequences" in item and isinstance(item["sequences"], list):
                    # AntiFold-style: each item has a 'sequences' list of samples
                    for sample in item["sequences"]:
                        if isinstance(sample, dict):
                            heavy = sample.get("heavy")
                            light = sample.get("light")
                            if heavy and light:
                                seq = f"{heavy}:{light}"
                            elif heavy:
                                seq = heavy
                            else:
                                seq = sample.get("sequence", "")
                            if seq:
                                sequences.append(
                                    {
                                        "sequence": seq,
                                        "heavy_chain": heavy,
                                        "light_chain": light,
                                        "score": sample.get("score"),
                                        "global_score": sample.get("global_score"),
                                        "seq_recovery": sample.get("seq_recovery"),
                                    }
                                )
                else:
                    # MPNN / flat dict
                    seq = item.get("sequence", "")
                    if seq:
                        sequences.append({"sequence": seq})
        return sequences

    @staticmethod
    def _create_masked_sequence(
        sequence: str,
        mask_positions: Union[str, List[int]],
        mask_fraction: float = 0.15,
        mask_token: str = "<mask>",
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

        if mask_positions == "auto":
            # Randomly select positions to mask
            num_to_mask = max(1, int(len(sequence) * mask_fraction))
            positions = random.sample(range(len(sequence)), num_to_mask)
        else:
            positions = mask_positions

        # Apply masking
        for pos in positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = mask_token

        masked_seq = "".join(seq_list)
        return masked_seq, sorted(positions)

    @staticmethod
    def _is_masked_lm(model_name: str) -> bool:
        """Check if a model is a masked language model."""
        mlm_models = ["esm", "esm1v", "esm2", "esm1b"]
        return any(mlm in model_name.lower() for mlm in mlm_models)

    async def _run_remasking(
        self,
        config: RemaskingConfig,
        parent_sequence: str,
        num_variants: int,
    ) -> List[Dict]:
        """Run MLM remasking generation using RemaskingConfig directly."""
        print(
            f"  {config.model_name} (remasking): generating {num_variants} variants..."
        )
        api = None
        try:
            api = BioLMApiClient(config.model_name)
            remasker = MLMRemasker(config, api_client=api, model_name=config.model_name)
            variants = await remasker.generate_variants(
                parent_sequence,
                num_variants=num_variants,
                deduplicate=True,
            )
            return [
                {
                    "sequence": seq,
                    "model_name": config.model_name,
                    "temperature": config.temperature,
                    "sampling_params": {},
                    "generation_method": "remask",
                    "parent_sequence": parent_sequence,
                    "num_mutations": meta.get("num_mutations"),
                    "mutation_rate": meta.get("mutation_rate"),
                }
                for seq, meta in variants
            ]
        finally:
            if api:
                await api.shutdown()

    async def _run_direct_generation(
        self,
        config: DirectGenerationConfig,
        datastore: DataStore,
        df_input: pd.DataFrame,
    ) -> List[Dict]:
        """Run generation using DirectGenerationConfig.

        The caller is responsible for setting the correct ``item_field`` and
        ``params`` for the target model.  No auto-detection is performed.
        """
        # ---- Resolve input values ----------------------------------------
        if config.structure_column and config.structure_column in df_input.columns:
            input_values = df_input[config.structure_column].dropna().tolist()
        elif config.structure_path:
            import tempfile

            from biolmai.pipeline.utils import cif_to_pdb, load_structure_string

            fmt, structure_str = load_structure_string(config.structure_path)
            if fmt == "cif":
                import os

                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                    pdb_path = f.name
                try:
                    cif_to_pdb(config.structure_path, pdb_path)
                    _, structure_str = load_structure_string(pdb_path)
                finally:
                    if os.path.exists(pdb_path):
                        os.remove(pdb_path)
            input_values = [structure_str]
        elif config.sequence is not None:
            input_values = [config.sequence]
        else:
            raise ValueError(
                "DirectGenerationConfig requires structure_path, structure_column, or sequence"
            )

        # ---- Params: user dict takes priority; fallback to convenience fields
        params: Dict[str, Any] = (
            config.params
            if config.params
            else {
                "num_sequences": config.num_sequences,
                "temperature": config.temperature,
            }
        )

        # ---- Generate ----------------------------------------------------
        results: List[Dict] = []
        api = None
        try:
            api = BioLMApiClient(config.model_name)
            for input_value in input_values:
                print(f"  {config.model_name} (direct): generating sequences...")
                items = [{config.item_field: input_value}]
                raw = await api.generate(items=items, params=params)
                if isinstance(raw, dict) and "error" in raw:
                    raise ValueError(f"API error from {config.model_name}: {raw.get('error')}")
                for seq_data in self._extract_sequences(raw):
                    results.append(
                        {
                            "model_name": config.model_name,
                            "temperature": params.get(
                                "temperature", config.temperature
                            ),
                            "sampling_params": params,
                            "generation_method": "direct",
                            "parent_sequence": config.sequence,
                            **seq_data,
                        }
                    )
        finally:
            if api:
                await api.shutdown()

        return results

    async def _generate_with_config(
        self, config: GenerationConfig, datastore: DataStore
    ) -> List[Dict]:
        """
        Generate sequences with a single configuration.

        Returns:
            List of dicts with keys: sequence, model_name, temperature, sampling_params
        """
        results = []

        # Determine if we need temperature scanning
        temps = (
            [config.temperature]
            if isinstance(config.temperature, (int, float))
            else config.temperature
        )

        for temp in temps:
            print(
                f"  {config.model_name} @ T={temp}: generating {config.num_sequences} sequences..."
            )

            # Create async API client
            api = None
            try:
                api = BioLMApiClient(config.model_name)
                if config.generation_method == "remask" or (
                    config.generation_method == "auto"
                    and self._is_masked_lm(config.model_name)
                ):
                    # Remasking approach for MLMs
                    if config.parent_sequence is None:
                        raise ValueError("parent_sequence required for remasking")

                    # Create RemaskingConfig from GenerationConfig
                    remask_config = RemaskingConfig(
                        model_name=config.model_name,  # Use model from GenerationConfig
                        mask_fraction=config.mask_fraction,
                        mask_positions=config.mask_positions,
                        num_iterations=config.sampling_params.get("num_iterations", 1),
                        temperature=temp,
                        top_k=config.sampling_params.get("top_k"),
                        top_p=config.sampling_params.get("top_p"),
                        conserved_positions=config.sampling_params.get(
                            "conserved_positions"
                        ),
                        mask_strategy=config.sampling_params.get(
                            "mask_strategy", "random"
                        ),
                    )

                    # Create remasker
                    remasker = MLMRemasker(
                        remask_config, api_client=api, model_name=config.model_name
                    )

                    # Generate variants (now async)
                    variants = await remasker.generate_variants(
                        config.parent_sequence,
                        num_variants=config.num_sequences,
                        deduplicate=True,
                    )

                    # Store results
                    for seq, metadata in variants:
                        results.append(
                            {
                                "sequence": seq,
                                "model_name": config.model_name,
                                "temperature": temp,
                                "sampling_params": config.sampling_params,
                                "generation_method": "remask",
                                "parent_sequence": config.parent_sequence,
                                "num_mutations": metadata.get("num_mutations"),
                                "mutation_rate": metadata.get("mutation_rate"),
                            }
                        )

                else:
                    # Direct generation (for inherently generative models)
                    params = {
                        **config.sampling_params,
                        "temperature": temp,
                        "num_sequences": config.num_sequences,
                    }

                    # Prepare items
                    if config.parent_sequence:
                        items = [{"sequence": config.parent_sequence}]
                    else:
                        items = [{}]  # Unconditional generation

                    # Generate (now async)
                    result = await api.generate(items=items, params=params)

                    # Check for API errors
                    if isinstance(result, dict) and "error" in result:
                        raise ValueError(f"API error from {config.model_name}: {result.get('error')}")

                    # Extract sequences
                    if isinstance(result, list):
                        for item in result:
                            if isinstance(item, dict):
                                seq = item.get("sequence", str(item))
                            else:
                                seq = str(item)

                            results.append(
                                {
                                    "sequence": seq,
                                    "model_name": config.model_name,
                                    "temperature": temp,
                                    "sampling_params": config.sampling_params,
                                    "generation_method": "generate",
                                    "parent_sequence": config.parent_sequence,
                                }
                            )

            finally:
                if api:
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
            parent = getattr(cfg, "parent_sequence", None)
            num = getattr(cfg, "num_variants", 100)
            if parent is None:
                raise ValueError("RemaskingConfig requires a parent_sequence attribute")
            return await self._run_remasking(
                cfg, parent_sequence=parent, num_variants=num
            )
        elif isinstance(cfg, DirectGenerationConfig):
            return await self._run_direct_generation(cfg, datastore, df_input)
        elif isinstance(cfg, GenerationConfig):
            return await self._generate_with_config(cfg, datastore)
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Generate sequences using configured models."""

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._dispatch_config(self.configs[0], datastore, df)
        else:
            tasks = [self._dispatch_config(cfg, datastore, df) for cfg in self.configs]
            results_list = await asyncio.gather(*tasks)
            results = [item for sublist in results_list for item in sublist]

        df_generated = (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                columns=[
                    "sequence",
                    "model_name",
                    "temperature",
                    "sampling_params",
                    "generation_method",
                    "parent_sequence",
                ]
            )
        )
        initial_count = len(df_generated)

        if self.deduplicate and initial_count > 0:
            df_generated = df_generated.drop_duplicates(
                subset=["sequence"]
            ).reset_index(drop=True)
            deduplicated_count = initial_count - len(df_generated)
            if deduplicated_count > 0:
                print(
                    f"  Deduplicated: {deduplicated_count} sequences ({len(df_generated)} unique)"
                )

        if len(df_generated) > 0:
            print(f"  Adding {len(df_generated)} sequences to datastore...")

            # Batch-insert all sequences in one vectorized call
            seq_ids = datastore.add_sequences_batch(df_generated["sequence"].tolist())
            df_generated["sequence_id"] = seq_ids

            # Store generation metadata per sequence
            for row_idx, (seq_id, row) in enumerate(
                zip(seq_ids, df_generated.itertuples(index=False))
            ):
                sampling_params = getattr(row, "sampling_params", None) or {}
                datastore.add_generation_metadata(
                    seq_id,
                    model_name=row.model_name,
                    temperature=getattr(row, "temperature", None),
                    top_k=sampling_params.get("top_k"),
                    top_p=sampling_params.get("top_p"),
                    num_return_sequences=sampling_params.get("num_return_sequences"),
                    do_sample=sampling_params.get("do_sample"),
                    repetition_penalty=sampling_params.get("repetition_penalty"),
                    max_length=sampling_params.get("max_length"),
                )

        return df_generated, StageResult(
            stage_name=self.name,
            input_count=0,
            output_count=len(df_generated),
            computed_count=initial_count,
            metadata={
                "deduplicated": (
                    initial_count - len(df_generated) if self.deduplicate else 0
                )
            },
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
        generation_configs: Optional[
            List[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]
        ] = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.generation_configs = generation_configs or []
        self.deduplicate = deduplicate

        # Add generation stage automatically
        if self.generation_configs:
            gen_stage = GenerationStage(
                name="generation",
                configs=self.generation_configs,
                deduplicate=self.deduplicate,
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
        if "generation" in self._stage_data:
            return self._stage_data["generation"]

        # If no generation configs, return empty DataFrame
        return pd.DataFrame(columns=["sequence", "sequence_id"])

    def add_generation_config(self, config: GenerationConfig):
        """Add a generation configuration."""
        self.generation_configs.append(config)

        # Update generation stage if it exists
        if self.stages and isinstance(self.stages[0], GenerationStage):
            self.stages[0].configs = self.generation_configs
        else:
            # Create generation stage
            gen_stage = GenerationStage(
                name="generation",
                configs=self.generation_configs,
                deduplicate=self.deduplicate,
            )
            self.stages.insert(0, gen_stage)

        return self

    def add_prediction(
        self,
        model_name: str,
        action: str = "predict",
        prediction_type: Optional[str] = None,
        params: Optional[Dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs,
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
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_predictions(
        self,
        models: List[Union[str, Dict]],
        action: str = "predict",
        depends_on: Optional[List[str]] = None,
        **kwargs,
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

        for model in models:
            if isinstance(model, str):
                # Simple model name
                self.add_prediction(
                    model_name=model, action=action, depends_on=depends_on, **kwargs
                )
            elif isinstance(model, dict):
                # Dict with config
                model_config = {**kwargs, **model}
                model_config["depends_on"] = model_config.get("depends_on", depends_on)

                model_name = model_config.pop("model_name")
                self.add_prediction(model_name=model_name, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")

        return self

    def add_filter(
        self,
        filter_func,
        stage_name: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        **kwargs,
    ):
        """Add a filter stage (same as DataPipeline)."""
        from biolmai.pipeline.data import FilterStage

        stage_name = stage_name or f"filter_{len(self.stages)}"

        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on or [],
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_cofolding_prediction(
        self,
        model_name: str,
        action: str = "predict",
        stage_name: Optional[str] = None,
        prediction_type: str = "structure",
        sequence_chain_id: str = "A",
        sequence_entity_type: str = "protein",
        static_entities: Optional[List["FoldingEntity"]] = None,
        depends_on: Optional[List[str]] = None,
        params: Optional[Dict] = None,
        batch_size: int = 1,
    ):
        """
        Add a co-folding prediction stage (Boltz2, Chai-1).

        Each sequence in the pipeline is treated as the primary protein chain.
        ``static_entities`` lets you inject additional molecules — ligands,
        cofactors, DNA/RNA strands, or extra protein chains — that are held
        constant across all designs.

        Args:
            model_name: BioLM model slug, e.g. ``'boltz2'`` or ``'chai1'``.
            action: API action (default ``'predict'``).
            stage_name: Stage name (defaults to ``model_name``).
            prediction_type: Column name for the confidence score (default
                ``'structure'``).
            sequence_chain_id: Chain ID assigned to the pipeline's primary
                sequence in the multi-molecule request (e.g. ``'A'``).
            sequence_entity_type: Molecule type for the primary sequence:
                ``'protein'``, ``'dna'``, or ``'rna'`` (default ``'protein'``).
            static_entities: List of :class:`FoldingEntity` objects — ligands,
                cofactors, extra proteins/DNA/RNA — included in every request.
            depends_on: Upstream stage names this stage waits for.
            params: Model-specific params dict (e.g.
                ``{'recycling_steps': 3, 'sampling_steps': 20}`` for Boltz).
            batch_size: Sequences per API request (default 1; co-folding models
                are typically batch-size-1).

        Example::

            pipeline.add_cofolding_prediction(
                model_name='boltz2',
                static_entities=[
                    FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
                    FoldingEntity(id='B', entity_type='protein',
                                  sequence='MKTAYIAKQRQ'),
                ],
                depends_on=['filter_top50'],
            )
        """
        from biolmai.pipeline.data import CofoldingPredictionStage

        stage_name = stage_name or model_name
        stage = CofoldingPredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            prediction_type=prediction_type,
            sequence_chain_id=sequence_chain_id,
            sequence_entity_type=sequence_entity_type,
            static_entities=static_entities or [],
            params=params or {},
            batch_size=batch_size,
            depends_on=depends_on or [],
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
                df_generated, result = await gen_stage.process(
                    pd.DataFrame(), self.datastore
                )
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
    **kwargs,
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
        parent_sequence=parent_sequence,
    )

    pipeline = GenerativePipeline(generation_configs=[config], **kwargs)
    pipeline.run()
    return pipeline.get_final_data()
