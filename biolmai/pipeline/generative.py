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
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from biolmai.client import BioLMApiClient  # Use async client
from biolmai.pipeline.base import BasePipeline, Stage, StageResult, WorkingSet
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
    params: dict[str, Any] = field(default_factory=dict)
    # Simple fallbacks used only when params is empty
    num_sequences: int = 100
    temperature: float = 1.0
    # Read structures from the DuckDB structures table (set by an upstream stage)
    structure_from_stage: Optional[str] = None
    structure_from_model: Optional[str] = None
    # Run the same generation call n_runs times in parallel.
    # Total output = (sequences per call) × n_runs, then deduped.
    n_runs: int = 1


    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        return {
            "type": "DirectGenerationConfig",
            "model_name": self.model_name,
            "structure_path": self.structure_path,
            "structure_column": self.structure_column,
            "sequence": self.sequence,
            "item_field": self.item_field,
            "params": self.params,
            "num_sequences": self.num_sequences,
            "temperature": self.temperature,
            "structure_from_stage": self.structure_from_stage,
            "structure_from_model": self.structure_from_model,
            "n_runs": self.n_runs,
        }


@dataclass
class SequenceSourceConfig:
    """Use existing sequences as the generation-slot source — no API calls made.

    Plug into ``set_generation()`` (or use ``pipeline.use_sequences()``) to
    feed existing data through prediction/filter stages without generating new
    sequences.  The provided sequences are added to the DuckDB via the normal
    dedup path, so sequences already present just return their existing IDs.

    Args:
        sequences: Source of sequences — one of:

            * ``list[str]``: plain amino-acid strings
            * ``pd.DataFrame``: must contain *column* (default ``"sequence"``)
            * ``str`` / ``Path``: path to a CSV or FASTA (``.fasta``/``.fa``) file
            * ``None``: reload all sequences already in the DuckDB (requires
              ``from_db=True`` OR leaving *sequences* as None)

        column: Column name when *sequences* is a DataFrame or CSV
            (default ``"sequence"``).
        from_db: Pull all sequences already present in the pipeline's DuckDB
            instead of loading new ones.  Equivalent to ``sequences=None``.

    Example::

        # Inject a list
        pipeline.use_sequences(["MKTAY", "MKLLIV"]).run()

        # Use all sequences already in the DB (e.g. after from_db() recovery)
        pipeline.use_sequences(from_db=True).run()

        # Load from CSV
        pipeline.use_sequences("candidates.csv").run()
    """

    sequences: Optional[Union[list, "pd.DataFrame", str, Path]] = None
    column: str = "sequence"
    from_db: bool = False

    def to_spec(self) -> dict:
        """Serialize for pipeline definition persistence.

        Live DataFrames / arbitrary paths are not serializable — on reconstruct
        we fall back to ``from_db=True`` so the existing DB sequences are reused.
        """
        return {
            "type": "SequenceSourceConfig",
            "from_db": True,
            "column": self.column,
        }


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
    temperature: Union[float, list[float]] = 1.0
    sampling_params: dict[str, Any] = field(default_factory=dict)
    generation_method: str = "generate"  # 'generate' or 'remask'
    parent_sequence: Optional[str] = None
    mask_positions: Union[str, list[int]] = "auto"
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
            list[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]
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

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        configs_specs = []
        for cfg in self.configs:
            if hasattr(cfg, "to_spec"):
                configs_specs.append(cfg.to_spec())
            else:
                raise NotImplementedError(
                    f"GenerationStage '{self.name}': config type "
                    f"'{type(cfg).__name__}' does not implement to_spec(). "
                    "Only RemaskingConfig and DirectGenerationConfig are serializable. "
                    "Migrate away from GenerationConfig (deprecated) to enable from_db() recovery."
                )
        return {
            "type": "GenerationStage",
            "name": self.name,
            "configs": configs_specs,
            "deduplicate": self.deduplicate,
            "depends_on": self.depends_on,
        }

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sequences(raw: Any) -> list[dict]:
        """
        Extract sequence dicts from any BioLM generation response format.

        Handles:
        - **Flat list** (MPNN): ``[{sequence, pdb, ...}, ...]``
        - **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
        - **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``
        """
        if not raw:
            return []
        sequences: list[dict] = []
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
        mask_positions: Union[str, list[int]],
        mask_fraction: float = 0.15,
        mask_token: str = "<mask>",
    ) -> tuple[str, list[int]]:
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

    async def _run_remasking(
        self,
        config: RemaskingConfig,
        parent_sequence: str,
        num_variants: int,
    ) -> list[dict]:
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
        context=None,
    ) -> list[dict]:
        """Run generation using DirectGenerationConfig.

        The caller is responsible for setting the correct ``item_field`` and
        ``params`` for the target model.  No auto-detection is performed.
        """
        # ---- Resolve input values ----------------------------------------
        if config.structure_from_stage or config.structure_from_model:
            # Read structures from DuckDB (set by an upstream prediction stage)
            model_filter = config.structure_from_model
            if model_filter:
                seq_ids_rows = datastore.conn.execute(
                    "SELECT sequence_id FROM structures WHERE model_name = ?",
                    [model_filter],
                ).fetchall()
            else:
                seq_ids_rows = datastore.conn.execute(
                    "SELECT sequence_id FROM structures"
                ).fetchall()
            struct_seq_ids = [r[0] for r in seq_ids_rows]
            if not struct_seq_ids:
                raise ValueError(
                    f"No structures found in datastore"
                    f"{' for model ' + model_filter if model_filter else ''}"
                )
            structs_df = datastore.get_structures_bulk(struct_seq_ids)
            if structs_df.empty:
                raise ValueError(
                    f"No structures found in datastore"
                    f"{' for model ' + model_filter if model_filter else ''}"
                )
            input_values = structs_df["structure_str"].dropna().tolist()
        elif config.structure_column and config.structure_column in df_input.columns:
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
        params: dict[str, Any] = (
            config.params
            if config.params
            else {
                "num_sequences": config.num_sequences,
                "temperature": config.temperature,
            }
        )

        # ---- Generate ----------------------------------------------------
        n_runs = getattr(config, "n_runs", 1) or 1
        results: list[dict] = []
        api = None
        try:
            api = BioLMApiClient(config.model_name)
            for input_value in input_values:
                if n_runs > 1:
                    print(
                        f"  {config.model_name} (direct): generating sequences "
                        f"({n_runs} parallel runs)..."
                    )
                else:
                    print(f"  {config.model_name} (direct): generating sequences...")
                items = [{config.item_field: input_value}]

                # Run generation n_runs times in parallel
                async def _single_run():
                    return await api.generate(items=items, params=params)

                if n_runs > 1:
                    raw_list = await asyncio.gather(
                        *[_single_run() for _ in range(n_runs)]
                    )
                else:
                    raw_list = [await api.generate(items=items, params=params)]

                for raw in raw_list:
                    if isinstance(raw, dict) and "error" in raw:
                        raise ValueError(
                            f"API error from {config.model_name}: {raw.get('error')}"
                        )
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
    ) -> list[dict]:
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
                if config.generation_method == "remask":
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
                        raise ValueError(
                            f"API error from {config.model_name}: {result.get('error')}"
                        )

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

    @staticmethod
    async def _run_sequence_source(
        config: SequenceSourceConfig, datastore: DataStore
    ) -> list[dict]:
        """Load sequences from an existing source without calling any API.

        Handles list[str], pd.DataFrame, CSV/FASTA paths, and DB reload.
        """
        sequences: list[str] = []

        if config.from_db or config.sequences is None:
            df_db = datastore.get_all_sequences()
            sequences = df_db["sequence"].dropna().tolist() if not df_db.empty else []

        elif isinstance(config.sequences, list):
            sequences = [str(s) for s in config.sequences if s]

        elif isinstance(config.sequences, pd.DataFrame):
            col = config.column
            if col not in config.sequences.columns:
                raise ValueError(
                    f"SequenceSourceConfig: column '{col}' not found in DataFrame. "
                    f"Available: {list(config.sequences.columns)}"
                )
            sequences = config.sequences[col].dropna().tolist()

        elif isinstance(config.sequences, (str, Path)):
            path = Path(config.sequences)
            if path.suffix.lower() in (".fasta", ".fa", ".faa"):
                # Minimal FASTA parser — no external deps
                seqs: list[str] = []
                current: list[str] = []
                with open(path) as fh:
                    for line in fh:
                        line = line.rstrip()
                        if line.startswith(">"):
                            if current:
                                seqs.append("".join(current))
                                current = []
                        else:
                            current.append(line)
                    if current:
                        seqs.append("".join(current))
                sequences = [s for s in seqs if s]
            else:
                df_csv = pd.read_csv(path)
                col = config.column
                if col not in df_csv.columns:
                    raise ValueError(
                        f"SequenceSourceConfig: column '{col}' not found in '{path}'. "
                        f"Available: {list(df_csv.columns)}"
                    )
                sequences = df_csv[col].dropna().tolist()

        else:
            raise TypeError(
                f"SequenceSourceConfig.sequences must be list, DataFrame, file path, or None. "
                f"Got {type(config.sequences)}"
            )

        return [
            {
                "sequence": s,
                "model_name": "data_source",
                "temperature": None,
                "sampling_params": {},
                "generation_method": "data_source",
                "parent_sequence": None,
            }
            for s in sequences
            if s
        ]

    async def _dispatch_config(
        self,
        cfg: Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig, SequenceSourceConfig],
        datastore: DataStore,
        df_input: pd.DataFrame,
        context=None,
    ) -> list[dict]:
        """Dispatch a single config to the appropriate generation method."""
        if isinstance(cfg, SequenceSourceConfig):
            return await self._run_sequence_source(cfg, datastore)
        elif isinstance(cfg, RemaskingConfig):
            parent = getattr(cfg, "parent_sequence", None)
            num = getattr(cfg, "num_variants", 100)
            if parent is None:
                raise ValueError("RemaskingConfig requires a parent_sequence attribute")
            return await self._run_remasking(
                cfg, parent_sequence=parent, num_variants=num
            )
        elif isinstance(cfg, DirectGenerationConfig):
            return await self._run_direct_generation(
                cfg, datastore, df_input, context=context
            )
        elif isinstance(cfg, GenerationConfig):
            return await self._generate_with_config(cfg, datastore)
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Generate sequences using configured models."""
        context = kwargs.get("context")

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._dispatch_config(
                self.configs[0], datastore, df, context=context
            )
        else:
            tasks = [
                self._dispatch_config(cfg, datastore, df, context=context)
                for cfg in self.configs
            ]
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
            for _row_idx, (seq_id, row) in enumerate(
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

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Generate sequences and return a WorkingSet of the new IDs."""
        df_generated, result = await self.process(pd.DataFrame(), datastore, **kwargs)
        if "sequence_id" in df_generated.columns:
            return WorkingSet.from_ids(df_generated["sequence_id"].tolist()), result
        return WorkingSet(frozenset()), result


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
        >>> pipeline.add_prediction('esmfold', extractions='mean_plddt', columns='plddt')
        >>> results = pipeline.run()
    """

    def __init__(
        self,
        generation_configs: Optional[
            list[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig]]
        ] = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.generation_configs = generation_configs or []
        self.deduplicate = deduplicate
        self._generated_ws = None  # set by run_async() after generation stage

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generation_slot(self) -> Optional[GenerationStage]:
        """Return the single GenerationStage at slot-0, or None."""
        if self.stages and isinstance(self.stages[0], GenerationStage):
            return self.stages[0]
        return None

    def _set_generation_slot(self, stage: GenerationStage) -> None:
        """Replace slot-0 with *stage*, preserving all downstream stages."""
        self.stages = [s for s in self.stages if not isinstance(s, GenerationStage)]
        self.stages.insert(0, stage)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_stage(self, stage: Stage) -> None:
        """Add a stage to the pipeline.

        If *stage* is a :class:`GenerationStage` it **replaces** the current
        generation slot (there is always exactly one, at position 0) rather
        than appending.  All other stage types are appended normally.
        """
        if isinstance(stage, GenerationStage):
            self._set_generation_slot(stage)
        else:
            super().add_stage(stage)

    def set_generation(
        self,
        *configs: Union[RemaskingConfig, DirectGenerationConfig],
        stage_name: str = "generation",
        deduplicate: bool = True,
    ) -> "GenerativePipeline":
        """Set (or replace) the generation slot with one or more configs.

        Multiple configs run in parallel — use this for multi-model generation
        or temperature scanning.  Every call to ``.run()`` re-runs generation
        from scratch; downstream stages use their prediction cache so only
        truly new sequences are computed.

        Args:
            *configs: One or more :class:`RemaskingConfig` or
                :class:`DirectGenerationConfig` objects.
            stage_name: Name for the generation stage (default ``"generation"``).
            deduplicate: Deduplicate across all configs (default True).

        Returns:
            Self, for method chaining.

        Example::

            # Single model
            pipeline.set_generation(
                DirectGenerationConfig("dsm-150m-base", sequence=parent, num_sequences=200)
            ).run()

            # Two models in parallel — sequences from both trickle through the funnel
            pipeline.set_generation(
                DirectGenerationConfig("dsm-150m-base", sequence=parent, num_sequences=100),
                RemaskingConfig("esm-150m", mask_fraction=0.15),
            ).run()
        """
        if not configs:
            raise ValueError("set_generation() requires at least one config")
        new_stage = GenerationStage(
            name=stage_name,
            configs=list(configs),
            deduplicate=deduplicate,
        )
        self._set_generation_slot(new_stage)
        return self

    # Keep replace_generation as a convenience alias for single-config swaps
    def replace_generation(
        self,
        config: Union[RemaskingConfig, DirectGenerationConfig],
        stage_name: Optional[str] = None,
        deduplicate: bool = True,
    ) -> "GenerativePipeline":
        """Swap the generation slot with a single new config.

        Equivalent to ``set_generation(config, stage_name=stage_name)``.
        Kept for backwards compatibility and single-config convenience.
        """
        existing = self._generation_slot()
        name = stage_name or (existing.name if existing else "generation")
        return self.set_generation(config, stage_name=name, deduplicate=deduplicate)

    def use_sequences(
        self,
        sequences=None,
        column: str = "sequence",
        stage_name: str = "data_source",
        from_db: bool = False,
    ) -> "GenerativePipeline":
        """Use existing sequences as the pipeline source instead of generating.

        Replaces the generation slot with a :class:`SequenceSourceConfig`.
        Downstream prediction and filter stages run on the provided sequences,
        using the normal DuckDB prediction cache for anything already computed.

        Args:
            sequences: One of:

                * ``list[str]`` — plain amino-acid strings
                * ``pd.DataFrame`` — must contain *column* (default ``"sequence"``)
                * ``str`` / ``Path`` — CSV or FASTA file
                * ``None`` — reload all sequences already in the DuckDB

            column: Column name when *sequences* is a DataFrame or CSV.
            stage_name: Name for the source stage (default ``"data_source"``).
            from_db: Pull all sequences already present in this pipeline's DuckDB.

        Returns:
            Self, for method chaining.

        Example::

            # Inject a list
            pipeline.use_sequences(["MKTAY", "MKLLIV"]).run()

            # Use all sequences already in the DB (e.g. recover + rerun)
            pipeline.use_sequences(from_db=True).run()

            # Load from CSV and run through the existing filter/predict stages
            pipeline.use_sequences("candidates.csv").run()
        """
        return self.set_generation(
            SequenceSourceConfig(sequences=sequences, column=column, from_db=from_db),
            stage_name=stage_name,
        )

    def add_generation_config(
        self, config: Union[RemaskingConfig, DirectGenerationConfig]
    ) -> "GenerativePipeline":
        """Append a config to the existing generation slot.

        Use this to add a second model or temperature variant *alongside*
        the current generation config rather than replacing it.  If there is
        no generation slot yet, one is created.

        Args:
            config: Config to add.

        Returns:
            Self, for method chaining.
        """
        slot = self._generation_slot()
        if slot is not None:
            slot.configs.append(config)
        else:
            self._set_generation_slot(
                GenerationStage(
                    name="generation",
                    configs=[config],
                    deduplicate=self.deduplicate,
                )
            )
        return self

    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """Not used — GenerativePipeline overrides _get_initial_data_ws directly."""
        return pd.DataFrame(columns=["sequence", "sequence_id"])

    async def _get_initial_data_ws(self, **kwargs) -> WorkingSet:
        """Return the WorkingSet produced by the generation slot.

        ``_generated_ws`` is populated by ``run_async()`` after all
        GenerationStages have run; it is the union of every generated sequence ID.
        """
        if getattr(self, "_generated_ws", None) is not None:
            return self._generated_ws
        return WorkingSet(frozenset())

    def add_prediction(
        self,
        model_name: str,
        action: str = "predict",
        extractions=None,
        columns=None,
        params: Optional[dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """Add a prediction stage (same as DataPipeline)."""
        from biolmai.pipeline.data import PredictionStage

        if stage_name is None:
            if columns and isinstance(columns, str):
                stage_name = f"predict_{columns}"
            elif extractions and isinstance(extractions, str):
                stage_name = f"predict_{extractions}"
            else:
                stage_name = f"{model_name}_{action}"

        stage = PredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            extractions=extractions,
            columns=columns,
            params=params,
            depends_on=depends_on or [],
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_predictions(
        self,
        models: list[Union[str, dict]],
        action: str = "predict",
        depends_on: Optional[list[str]] = None,
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
            >>> pipeline.add_predictions(['temberture-regression', 'proteinmpnn', 'esm2'])
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
        depends_on: Optional[list[str]] = None,
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
        static_entities: Optional[list["FoldingEntity"]] = None,
        depends_on: Optional[list[str]] = None,
        params: Optional[dict] = None,
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

    async def run_async(self, **kwargs) -> dict[str, StageResult]:
        """
        Run the generative pipeline.

        All GenerationStages (regardless of position) are extracted and run first
        as sources — their outputs are unioned into the initial WorkingSet so
        generated sequences trickle through every downstream prediction/filter
        stage (the "funnel").  Idempotent: self.stages is always restored after
        execution.
        """
        import copy as _copy

        # Save original stages so the pipeline is idempotent (can be called again)
        original_stages = self.stages

        try:
            # Separate generation stages from processing stages (order-independent)
            gen_stages = [s for s in self.stages if isinstance(s, GenerationStage)]
            non_gen_stages = [s for s in self.stages if not isinstance(s, GenerationStage)]
            gen_names = {s.name for s in gen_stages}

            # Run every generation stage first, collect all produced sequence IDs
            all_gen_ids: set[int] = set()
            for gen_stage in gen_stages:
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Stage: {gen_stage.name} (Generation)")
                    print(f"Configs: {len(gen_stage.configs)}")
                    print(f"{'='*60}")

                df_generated, result = await gen_stage.process(
                    pd.DataFrame(), self.datastore
                )
                self.stage_results[gen_stage.name] = result
                self._stage_data[gen_stage.name] = df_generated

                if "sequence_id" in df_generated.columns:
                    ids = df_generated["sequence_id"].tolist()
                    ws = WorkingSet.from_ids(ids)
                    self._working_sets[gen_stage.name] = ws
                    all_gen_ids |= set(ids)

                if self.verbose:
                    print(f"\n{result}")
                    print(f"{'='*60}")

            # Union of all generated IDs becomes the initial WS for downstream stages
            self._generated_ws = WorkingSet.from_ids(list(all_gen_ids)) if all_gen_ids else None

            # Build the execution plan without the generation stages.
            # Strip gen stage names from depends_on so dependency resolution works.
            remaining_stages = []
            for s in non_gen_stages:
                s_copy = _copy.copy(s)
                s_copy.depends_on = [d for d in s.depends_on if d not in gen_names]
                remaining_stages.append(s_copy)
            self.stages = remaining_stages

            # Run remaining stages using base class (starts from _generated_ws)
            return await super().run_async(**kwargs)
        finally:
            # Always restore original stages so the pipeline remains idempotent
            self.stages = original_stages


# Convenience function for quick generation
def Generate(
    model_name: str,
    num_sequences: int = 100,
    temperature: Union[float, list[float]] = 1.0,
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
