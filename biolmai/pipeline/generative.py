"""
Generative pipeline for sequence generation using language models.

Supports:
- Masked language models (ESM, ESM-1v) with remasking
- Inherently generative models (ProteinMPNN, ProGen2, etc.)
- Temperature and sampling parameter scanning
- Multi-model generation in parallel
"""

from __future__ import annotations

import asyncio
import itertools as _itertools
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Module-level counter for unique temporary DuckDB table names
_GENERATIVE_TEMP_COUNTER = _itertools.count()

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

        Live DataFrames are not serializable — on reconstruct we fall back to
        ``from_db=True`` so the existing DB sequences are reused.  Plain
        ``list[str]`` sequences are included directly so from_db() reconstruction
        can replay the same input without requiring an existing DB (GEN-09 fix).
        """
        # GEN-09: include list[str] sequences when serializable
        if isinstance(self.sequences, list):
            return {
                "type": "SequenceSourceConfig",
                "sequences": self.sequences,
                "column": self.column,
                "from_db": False,
            }
        # DataFrames, Path objects, and None all fall back to from_db=True
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
        - **Single dict** (MPNN with unwrap_single=True): ``{sequence, pdb, ...}``
        - **Flat list** (MPNN, multiple results): ``[{sequence, pdb, ...}, ...]``
        - **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
        - **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``
        """
        # Normalize: BioLMApiClient with unwrap_single=True returns a bare dict
        # for single-item inputs. Wrap so the rest of the logic sees a uniform list.
        if isinstance(raw, dict):
            raw = [raw]
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
                        else:
                            # BUG-GEN-06 fix: log when a DSM sub-item has no sequence
                            logger.warning(
                                "Skipping generation result item with missing or empty "
                                "'sequence': %s",
                                str(sub)[:200],
                            )
            elif isinstance(item, dict):
                sequences_list = item.get("sequences", [])
                # GEN2-03: narrow AntiFold detection — require the first element
                # to be a dict with 'heavy' or 'light' key so that any other model
                # using a "sequences" key is not accidentally routed here.
                is_antifold = (
                    isinstance(sequences_list, list)
                    and len(sequences_list) > 0
                    and isinstance(sequences_list[0], dict)
                    and ("heavy" in sequences_list[0] or "light" in sequences_list[0])
                )
                if is_antifold:
                    # AntiFold-style: each item has a 'sequences' list of samples
                    for sample in sequences_list:
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
                                # BUG-GEN-06 fix: log when an AntiFold sample has no sequence
                                logger.warning(
                                    "Skipping generation result item with missing or empty "
                                    "'sequence': %s",
                                    str(sample)[:200],
                                )
                else:
                    # MPNN / flat dict
                    seq = item.get("sequence", "")
                    if seq:
                        sequences.append({"sequence": seq})
                    else:
                        # BUG-GEN-06 fix: log when a flat-dict item has no sequence
                        logger.warning(
                            "Skipping generation result item with missing or empty "
                            "'sequence': %s",
                            str(item)[:200],
                        )
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
        ws_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Run generation using DirectGenerationConfig.

        The caller is responsible for setting the correct ``item_field`` and
        ``params`` for the target model.  No auto-detection is performed.

        Args:
            ws_ids: Optional list of sequence IDs from the current working set.
                When provided, structure lookups are scoped to only these IDs to
                prevent cross-run contamination (GEN-05 fix).
        """
        # ---- Resolve input values ----------------------------------------
        # BUG-GEN-08 fix: warn clearly when structure_from_stage is set without
        # structure_from_model so the user knows all structures will be used.
        if config.structure_from_stage and not config.structure_from_model:
            warnings.warn(
                f"DirectGenerationConfig has 'structure_from_stage' set but "
                f"'structure_from_model' is not set. All structures from all upstream "
                f"models will be used, which may cause incorrect protein designs if "
                f"multiple structure-prediction stages have run. Set "
                f"structure_from_model='esmfold' (or your model name) to select the "
                f"correct structures.",
                UserWarning,
                stacklevel=3,
            )

        if config.structure_from_stage or config.structure_from_model:
            # GEN-05 fix: scope structure lookup to current working set sequence_ids.
            # Collect candidate IDs: prefer ws_ids, fall back to df_input sequence_ids,
            # then fall back to all sequences (unscoped).
            if ws_ids is not None:
                candidate_ws_ids = ws_ids
            elif not df_input.empty and "sequence_id" in df_input.columns:
                candidate_ws_ids = df_input["sequence_id"].tolist()
            else:
                candidate_ws_ids = None  # no scope available — use unscoped query

            # Read structures from DuckDB (set by an upstream prediction stage)
            model_filter = config.structure_from_model
            if candidate_ws_ids is not None:
                # Scoped query: only structures for sequences in the current WS
                ws_ids_df = pd.DataFrame({"sequence_id": candidate_ws_ids})
                _tmp_ws = f"_direct_gen_ws_{next(_GENERATIVE_TEMP_COUNTER)}"
                datastore.conn.register(_tmp_ws, ws_ids_df)
                try:
                    if model_filter:
                        seq_ids_rows = datastore.conn.execute(
                            f"SELECT st.sequence_id FROM structures st "
                            f"INNER JOIN {_tmp_ws} w ON st.sequence_id = w.sequence_id "
                            f"WHERE st.model_name = ?",
                            [model_filter],
                        ).fetchall()
                    else:
                        seq_ids_rows = datastore.conn.execute(
                            f"SELECT st.sequence_id FROM structures st "
                            f"INNER JOIN {_tmp_ws} w ON st.sequence_id = w.sequence_id"
                        ).fetchall()
                finally:
                    datastore.conn.unregister(_tmp_ws)
            elif model_filter:
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
                # GEN2-02: each concurrent run gets its own client to avoid
                # sharing a single api instance across concurrent coroutines.
                if n_runs > 1:
                    _model_name = config.model_name

                    async def _single_run(items=items, params=params):
                        _api = BioLMApiClient(_model_name)
                        try:
                            return await _api.generate(items=items, params=params)
                        finally:
                            await _api.shutdown()

                    _all_results = await asyncio.gather(
                        *[_single_run() for _ in range(n_runs)],
                        return_exceptions=True,
                    )
                    # BUG-GEN-07 fix: filter out exceptions so one failed run does
                    # not cancel all parallel runs.
                    raw_list = []
                    for _i, _r in enumerate(_all_results):
                        if isinstance(_r, Exception):
                            logger.warning(
                                "Generation run %d/%d failed: %s", _i + 1, n_runs, _r
                            )
                        else:
                            raw_list.append(_r)
                else:
                    raw_list = [await api.generate(items=items, params=params)]

                for raw in raw_list:
                    if isinstance(raw, dict) and "error" in raw:
                        raise ValueError(
                            f"API error from {config.model_name}: {raw.get('error')}"
                        )
                    extracted = self._extract_sequences(raw)
                    # GEN2-09: if extraction yielded nothing but raw is non-empty,
                    # check whether the first element is an error dict.
                    if not extracted and isinstance(raw, list) and raw and isinstance(raw[0], dict) and "error" in raw[0]:
                        raise ValueError(f"API returned error: {raw[0]}")
                    for seq_data in extracted:
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

                    # Extract sequences using the same format-aware extractor
                    # that handles flat/MPNN, AntiFold-nested, and DSM double-nested.
                    for extracted in GenerationStage._extract_sequences(result):
                        results.append(
                            {
                                **extracted,
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
        config: SequenceSourceConfig,
        datastore: DataStore,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """Load sequences from an existing source without calling any API.

        Handles list[str], pd.DataFrame, CSV/FASTA paths, and DB reload.

        When ``config.from_db=True`` and a ``run_id`` is provided, the query is
        scoped to sequences that have a ``generation_metadata`` row for that run
        (i.e. sequences generated in that run), preventing cross-run contamination
        on resume.  If no such rows exist (e.g. sequences were added via
        ``DataPipeline`` rather than generated), all DB sequences are returned.
        """
        sequences: list[str] = []

        if config.from_db or config.sequences is None:
            # GEN-02: scope to current run_id when resuming a GenerativePipeline run.
            # Only scope when this run has generation_metadata rows (i.e. sequences
            # were generated in this run).  Otherwise return all DB sequences to
            # preserve the use case of reloading externally-added sequences (e.g. from
            # a prior DataPipeline run loaded via use_sequences(from_db=True)).
            scoped = False
            if run_id:
                try:
                    count_row = datastore.conn.execute(
                        "SELECT COUNT(*) FROM generation_metadata WHERE run_id = ?",
                        [run_id],
                    ).fetchone()
                    if count_row and count_row[0] > 0:
                        rows = datastore.conn.execute(
                            """
                            SELECT s.sequence
                            FROM sequences s
                            INNER JOIN generation_metadata gm ON s.sequence_id = gm.sequence_id
                            WHERE gm.run_id = ?
                            """,
                            [run_id],
                        ).fetchall()
                        sequences = [r[0] for r in rows if r[0]]
                        scoped = True
                except Exception:
                    pass
            if not scoped:
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
        run_id: Optional[str] = None,
        ws_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Dispatch a single config to the appropriate generation method."""
        if isinstance(cfg, SequenceSourceConfig):
            return await self._run_sequence_source(cfg, datastore, run_id=run_id)
        elif isinstance(cfg, RemaskingConfig):
            # RemaskingConfig takes one parent → N variants.  The parent_sequence must
            # be supplied by the caller (set as a dynamic attribute on the config), because
            # accepting multiple parents from df_input would explode into ambiguous provenance:
            # which variant came from which parent?  Use one RemaskingConfig per parent.
            num = cfg.num_variants
            parent = cfg.parent_sequence
            if parent is None:
                raise ValueError(
                    "RemaskingConfig requires a parent_sequence. Set it before dispatch:\n"
                    "    config = RemaskingConfig(...)\n"
                    "    config.parent_sequence = 'MKTAYIAKQRQ'\n"
                    "    config.num_variants = 50\n"
                    "Or use GenerationConfig(remasking_config=..., parent_sequence=...) "
                    "for the legacy API."
                )
            return await self._run_remasking(cfg, parent_sequence=parent, num_variants=num)
        elif isinstance(cfg, DirectGenerationConfig):
            return await self._run_direct_generation(
                cfg, datastore, df_input, context=context, ws_ids=ws_ids
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
        run_id = kwargs.get("run_id")
        # GEN-05: extract WS IDs for structure scoping.
        # Prefer ws_ids passed explicitly from process_ws() so that structure
        # lookups are scoped to the current pipeline working set even when
        # process() is called with an empty DataFrame (the WorkingSet path).
        ws_ids: Optional[list[int]] = kwargs.get("ws_ids") or (
            df["sequence_id"].tolist()
            if not df.empty and "sequence_id" in df.columns
            else None
        )

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._dispatch_config(
                self.configs[0], datastore, df, context=context, run_id=run_id,
                ws_ids=ws_ids,
            )
        else:
            tasks = [
                self._dispatch_config(
                    cfg, datastore, df, context=context, run_id=run_id, ws_ids=ws_ids,
                )
                for cfg in self.configs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for i, r in enumerate(results_list):
                if isinstance(r, Exception):
                    logger.warning(
                        "Generation config %d/%d failed: %s",
                        i + 1, len(self.configs), r,
                    )
                else:
                    results.extend(r)

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

        # BUG-GEN-05 fix: warn when generation produces 0 sequences so the user
        # gets an actionable message rather than silently passing nothing downstream.
        if df_generated.empty:
            warnings.warn(
                f"GenerationStage '{self.name}' produced 0 sequences. "
                "Check API response and generation config.",
                UserWarning,
                stacklevel=2,
            )

        if self.deduplicate and initial_count > 0:
            # GEN2-06: normalize to uppercase before dedup so case variants
            # (e.g. 'mktay' vs 'MKTAY') are treated as identical.
            df_generated["sequence"] = df_generated["sequence"].str.upper()
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

            # Detect extra columns from generation (e.g. heavy_chain, light_chain
            # from AntiFold) and store them on the sequences table so they survive
            # WorkingSet materialization.
            _gen_extra_cols = [
                c for c in ("heavy_chain", "light_chain")
                if c in df_generated.columns
                and df_generated[c].notna().any()
            ]
            if _gen_extra_cols:
                datastore.ensure_input_columns(_gen_extra_cols)
                _all_cols = ["sequence"] + _gen_extra_cols
                seq_ids = datastore.add_sequences_batch(
                    input_df=df_generated[_all_cols],
                    input_columns=_all_cols,
                )
            else:
                # Batch-insert all sequences in one vectorized call
                seq_ids = datastore.add_sequences_batch(df_generated["sequence"].tolist())
            df_generated["sequence_id"] = seq_ids

            # Store numeric generation scores (global_score, score, seq_recovery)
            # as predictions so they survive WorkingSet materialization and are
            # available for downstream filters.
            _score_cols = [
                c for c in ("global_score", "score", "seq_recovery")
                if c in df_generated.columns
                and df_generated[c].notna().any()
            ]
            if _score_cols:
                pred_rows = []
                for sid, row in zip(seq_ids, df_generated.itertuples(index=False)):
                    for col in _score_cols:
                        val = getattr(row, col, None)
                        if val is not None:
                            pred_rows.append({
                                "sequence_id": sid,
                                "prediction_type": col,
                                "model_name": row.model_name,
                                "value": float(val),
                            })
                if pred_rows:
                    datastore.add_predictions_batch(pred_rows)

            # Store generation metadata in one batched INSERT (BUG-10 fix: replaces N+1 loop)
            run_id = kwargs.get("run_id", "")
            meta_rows = []
            for seq_id, row in zip(seq_ids, df_generated.itertuples(index=False)):
                sampling_params = getattr(row, "sampling_params", None) or {}
                meta_rows.append({
                    "sequence_id": seq_id,
                    "run_id": run_id,
                    "model_name": row.model_name,
                    "temperature": getattr(row, "temperature", None),
                    "sampling_params": sampling_params,
                })
            datastore.add_generation_metadata_batch(meta_rows)

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
        """Generate sequences and return a WorkingSet of the new IDs.

        The input WorkingSet's IDs are forwarded as ``ws_ids`` so that
        ``DirectGenerationConfig`` can scope structure lookups to sequences
        already in the current pipeline run (GEN-05 consistency: no cross-run
        contamination when ``structure_from_stage`` is set).
        """
        # Forward input WS IDs so structure lookups stay scoped to this run.
        if ws and "ws_ids" not in kwargs:
            kwargs["ws_ids"] = ws.to_list()
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
            slot.configs = list(slot.configs)
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
        if self._generated_ws is not None:
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

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = PredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            extractions=extractions,
            columns=columns,
            params=params,
            depends_on=depends_on,
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

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on,
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

        resume = kwargs.get("resume", False)

        # Reset generated_ws at start so previous run's data doesn't bleed through
        self._generated_ws = None

        # Save original stages so the pipeline is idempotent (can be called again)
        original_stages = self.stages

        # BUG-07 fix: create the pipeline_runs record BEFORE any generation stage runs,
        # so that mark_stage_complete() never writes orphan stage_completions rows.
        # create_pipeline_run() is idempotent (ON CONFLICT DO UPDATE), so the later
        # call inside super().run_async() is harmless.
        self.datastore.create_pipeline_run(
            run_id=self.run_id,
            pipeline_type=self.pipeline_type,
            config=self._get_config(),
            status="running",
        )

        try:
            # Separate generation stages from processing stages (order-independent)
            gen_stages = [s for s in self.stages if isinstance(s, GenerationStage)]
            non_gen_stages = [s for s in self.stages if not isinstance(s, GenerationStage)]
            gen_names = {s.name for s in gen_stages}

            # Run every generation stage first, collect all produced sequence IDs
            all_gen_ids: set[int] = set()
            for gen_stage in gen_stages:
                stage_id = f"{self.run_id}_{gen_stage.name}"

                # Resume: if this generation stage already completed, reload its
                # working set from generation_metadata rather than re-running.
                # generation_metadata.run_id scopes to this run, so the reload
                # is exact and won't pick up sequences from other runs.
                if resume and self.datastore:
                    sc = self.datastore.conn.execute(
                        "SELECT status, output_count FROM stage_completions WHERE stage_id = ?",
                        [stage_id],
                    ).fetchone()
                    if sc and sc[0] == "completed":
                        expected_count = sc[1]
                        ids = self.datastore.conn.execute(
                            "SELECT DISTINCT sequence_id FROM generation_metadata "
                            "WHERE run_id = ?",
                            [self.run_id],
                        ).df()["sequence_id"].tolist()
                        if ids:
                            if expected_count is not None and len(ids) != expected_count:
                                import warnings
                                warnings.warn(
                                    f"Stage '{gen_stage.name}': stage_completions recorded "
                                    f"{expected_count} sequences but only {len(ids)} found in "
                                    f"generation_metadata. Some sequences may be missing downstream "
                                    f"predictions and will be silently skipped. Consider re-running "
                                    f"without resume=True to regenerate.",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                            ws = WorkingSet.from_ids(ids)
                            self._working_sets[gen_stage.name] = ws
                            all_gen_ids |= set(ids)
                            if self.verbose:
                                print(f"\n{'='*60}")
                                print(f"✓ Stage '{gen_stage.name}' already complete "
                                      f"— reloaded {len(ids)} sequences from DB")
                                print(f"{'='*60}")
                            continue

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Stage: {gen_stage.name} (Generation)")
                    print(f"Configs: {len(gen_stage.configs)}")
                    print(f"{'='*60}")

                # BUG-03 fix: pass run_id and context so generation stages have
                # access to PipelineContext and resume bookkeeping.
                df_generated, result = await gen_stage.process(
                    pd.DataFrame(), self.datastore,
                    run_id=self.run_id, context=self.context,
                )
                self.stage_results[gen_stage.name] = result
                self._stage_data[gen_stage.name] = df_generated

                if "sequence_id" in df_generated.columns:
                    ids = df_generated["sequence_id"].tolist()
                    ws = WorkingSet.from_ids(ids)
                    self._working_sets[gen_stage.name] = ws
                    all_gen_ids |= set(ids)

                # Mark the generation stage complete so resume logic works
                if self.datastore:
                    self.datastore.mark_stage_complete(
                        run_id=self.run_id,
                        stage_name=gen_stage.name,
                        stage_id=stage_id,
                        input_count=0,
                        output_count=len(df_generated),
                        status="completed",
                    )

                if self.verbose:
                    print(f"\n{result}")
                    print(f"{'='*60}")

            # BUG-02 fix: warn when generation produced no sequences so the user
            # gets an actionable message rather than silently processing 0 inputs.
            if not all_gen_ids:
                import warnings
                warnings.warn(
                    "GenerativePipeline: no sequences were produced by any generation "
                    "stage. Downstream prediction/filter stages will receive 0 inputs. "
                    "Check your generation configs, API key, and model availability.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Union of all generated IDs becomes the initial WS for downstream stages
            self._generated_ws = WorkingSet.from_ids(list(all_gen_ids)) if all_gen_ids else None

            # BUG-GEN-09 fix: capture what THIS run generated before super().run_async()
            # resets internal dicts.  We restore it afterward so a second call to run()
            # always uses the fresh generation result, not a stale backup.
            _fresh_gen_ws = self._generated_ws

            # BUG-01 fix: save the full pipeline definition (including GenerationStage)
            # BEFORE stripping gen stages from self.stages.  super().run_async() will
            # call _save_definition_and_register_columns() again with remaining_stages
            # only — we capture the full definition_id here and restore it afterward
            # so that pipeline_runs.definition_id points to the complete definition.
            self._save_definition_and_register_columns()
            from biolmai.pipeline.pipeline_def import _pipeline_def_hash
            import json as _json
            _full_stages_specs = []
            for s in self.stages:
                try:
                    _full_stages_specs.append(s.to_spec())
                except NotImplementedError:
                    _full_stages_specs.append({"type": s.__class__.__name__, "name": s.name})
            _input_cols = self.input_schema.columns if self.input_schema else None
            _full_def_id = _pipeline_def_hash(self.pipeline_type, _input_cols, _full_stages_specs)

            # GEN2-01 fix: capture full_config_dict BEFORE the stage swap so
            # self._get_config() sees all stages (including generation stages).
            # This eliminates the concurrency hazard from the previous approach
            # of temporarily restoring original_stages inside the UPDATE block.
            full_config_dict = self._get_config()

            # Build the execution plan without the generation stages.
            # Strip gen stage names from depends_on so dependency resolution works.
            # BUG-13 fix: use deepcopy so _api_client and other mutable attrs are
            # not shared between the copy and the original across repeated run() calls.
            remaining_stages = []
            for s in non_gen_stages:
                s_copy = _copy.deepcopy(s)
                s_copy.depends_on = [d for d in s.depends_on if d not in gen_names]
                remaining_stages.append(s_copy)
            self.stages = remaining_stages

            # GEN-01 fix: backup gen stage dicts before super().run_async() wipes them.
            # BasePipeline.run_async() resets stage_results, _working_sets, _stage_data
            # at the top; we restore the generation stage entries afterward so callers
            # can inspect both generation and downstream results via stage_results.
            _gen_ws_backup = {
                s.name: self._working_sets[s.name]
                for s in gen_stages
                if s.name in self._working_sets
            }
            _gen_stage_backup = {
                s.name: self._stage_data[s.name]
                for s in gen_stages
                if s.name in self._stage_data
            }
            _gen_results_backup = {
                s.name: self.stage_results[s.name]
                for s in gen_stages
                if s.name in self.stage_results
            }

            # Run remaining stages using base class (starts from _generated_ws)
            run_result = await super().run_async(**kwargs)

            # BUG-GEN-09 fix: restore this run's fresh generation WS.
            # super().run_async() resets _generated_ws to None at the top of every
            # call, so on a second run() the old backup would overwrite the new
            # generation result.  Always prefer what this run actually generated.
            if _fresh_gen_ws is not None:
                self._generated_ws = _fresh_gen_ws

            # Restore generation stage entries wiped by super().run_async() reset
            for name, ws in _gen_ws_backup.items():
                if name not in self._working_sets:
                    self._working_sets[name] = ws
            for name, df_s in _gen_stage_backup.items():
                if name not in self._stage_data:
                    self._stage_data[name] = df_s
            for name, res in _gen_results_backup.items():
                if name not in self.stage_results:
                    self.stage_results[name] = res

            # Restore pipeline_runs.definition_id to the full definition (BUG-01 fix)
            # and update .config to include generation stages (GEN2-01 fix).
            _conn_alive = (
                self.datastore is not None
                and getattr(self.datastore, "conn", None) is not None
            )
            if _conn_alive:
                # GEN2-01: full_config was captured BEFORE the stage swap (see above),
                # so we use it directly here instead of doing a fragile stages swap.
                self.datastore.conn.execute(
                    "UPDATE pipeline_runs SET definition_id = ?, config = ? WHERE run_id = ?",
                    [_full_def_id, _json.dumps(full_config_dict), self.run_id],
                )

            return run_result
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
        temperature: Temperature (float) or list of temperatures for a temperature scan.
            When a list is given, one DirectGenerationConfig is created per temperature.
        parent_sequence: Optional parent sequence for sequence-conditioned models
        **kwargs: Additional arguments forwarded to GenerativePipeline

    Returns:
        DataFrame with generated sequences

    Example:
        >>> df = Generate('dsm-150m-base', num_sequences=100, parent_sequence='MKTAY')
    """
    # GEN2-07: validate that at least one of parent_sequence or structure_path is given
    if parent_sequence is None and "structure_path" not in kwargs:
        raise ValueError(
            "Generate() requires either parent_sequence= or structure_path= to be provided"
        )

    # BUG-API-04: support list of temperatures — create one config per temperature
    temps = temperature if isinstance(temperature, list) else [float(temperature)]
    item_field = "sequence" if parent_sequence is not None else "pdb"
    configs = [
        DirectGenerationConfig(
            model_name=model_name,
            sequence=parent_sequence,
            item_field=item_field,
            num_sequences=num_sequences,
            temperature=float(t),
        )
        for t in temps
    ]

    pipeline = GenerativePipeline(generation_configs=configs, **kwargs)
    # BUG-API-03: ensure DuckDB connection is closed even if run() raises
    try:
        pipeline.run()
        return pipeline.get_final_data()
    finally:
        if getattr(pipeline, "_auto_created_datastore", False):
            try:
                pipeline.datastore.close()
            except Exception:
                pass
