"""
Base Pipeline classes for stage management and execution.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSchema:
    """Describes the input columns for a pipeline.

    When set, these columns are the primary data — ``sequence`` is not required.
    Columns are stored directly on the ``sequences`` table via
    ``ALTER TABLE ADD COLUMN`` so that ``materialize_working_set()``, SQL
    filters, and ``item_columns`` all work via direct JOINs.

    Hashing uses all columns (sorted alphabetically) joined with ``\\x00``
    separators so that identical rows produce the same hash regardless of
    column order.

    Args:
        columns: List of column names that comprise the primary input.
    """

    columns: list[str]

    def hash_row(self, row: dict[str, str]) -> str:
        """SHA-256 hash of the row values across all input columns (sorted)."""
        import hashlib

        parts = [str(row.get(c, "")) for c in sorted(self.columns)]
        payload = "\x00".join(parts)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class PipelineContext:
    """Shared key-value store backed by DuckDB for inter-stage communication.

    Stages can read/write arbitrary data through the context. Common use
    case: stage 1 predicts structures (stored in the ``structures`` table),
    stage 2 reads them for structure-conditioned generation.

    Args:
        datastore: The pipeline's DuckDB datastore.
        run_id: Current pipeline run ID.
    """

    def __init__(self, datastore: DataStore, run_id: str):
        self._datastore = datastore
        self._run_id = run_id

    def set(self, key: str, value: Any):
        """Store a value in the pipeline context table."""
        self._datastore.set_context(self._run_id, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the pipeline context table."""
        val = self._datastore.get_context(self._run_id, key)
        return val if val is not None else default

    def get_structure(self, sequence_id: int, model_name: Optional[str] = None) -> Optional[dict]:
        """Convenience: fetch a structure from the datastore's structures table."""
        return self._datastore.get_structure(sequence_id, model_name)

    def get_structures_for_ws(
        self, ws: "WorkingSet", model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch structures for all sequences in a WorkingSet."""
        ids = list(ws.sequence_ids)
        df = self._datastore.get_structures_bulk(ids)
        if model_name and not df.empty:
            df = df[df["model_name"] == model_name]
        return df


@dataclass(frozen=True)
class WorkingSet:
    """Lightweight set of sequence IDs — replaces DataFrame as inter-stage transport.

    Stages operate on DuckDB directly and pass only the set of surviving
    sequence IDs to the next stage.  Materialization to DataFrame happens
    once at ``get_final_data()`` time.

    Memory: 1M IDs ≈ 28 MB (frozenset[int]) vs 500 MB+ DataFrame.
    """

    sequence_ids: frozenset[int]

    # --- convenience helpers ---------------------------------------------------

    def intersect(self, other: "WorkingSet") -> "WorkingSet":
        """Return a new WorkingSet containing only IDs present in both sets."""
        return WorkingSet(self.sequence_ids & other.sequence_ids)

    def union(self, other: "WorkingSet") -> "WorkingSet":
        """Return a new WorkingSet containing IDs from either set."""
        return WorkingSet(self.sequence_ids | other.sequence_ids)

    def difference(self, other: "WorkingSet") -> "WorkingSet":
        """Return IDs in *self* but not in *other*."""
        return WorkingSet(self.sequence_ids - other.sequence_ids)

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __bool__(self) -> bool:
        return bool(self.sequence_ids)

    def __iter__(self):
        return iter(self.sequence_ids)

    def __contains__(self, item: int) -> bool:
        return item in self.sequence_ids

    @classmethod
    def from_ids(cls, ids) -> "WorkingSet":
        """Create from any iterable of ints."""
        return cls(frozenset(int(i) for i in ids))

    def to_list(self) -> list[int]:
        """Return sorted list (useful for DuckDB queries)."""
        return sorted(self.sequence_ids)


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage_name: str
    input_count: int
    output_count: int
    filtered_count: int = 0
    cached_count: int = 0
    computed_count: int = 0
    elapsed_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"StageResult({self.stage_name}: "
            f"in={self.input_count}, out={self.output_count}, "
            f"cached={self.cached_count}, computed={self.computed_count}, "
            f"filtered={self.filtered_count}, time={self.elapsed_time:.1f}s)"
        )


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    A stage represents a single processing step in the pipeline.
    It can filter data, compute predictions, or transform sequences.

    Args:
        name: Stage name (must be unique within pipeline)
        cache_key: Unused collision-dedup key (auto-derived by PredictionStage)
        depends_on: List of stage names this stage depends on
        model_name: Model name for predictions/structures
        max_concurrent: Maximum concurrent API calls (for rate limiting)
    """

    def __init__(
        self,
        name: str,
        cache_key: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        model_name: Optional[str] = None,
        max_concurrent: int = 10,
    ):
        self.name = name
        self.cache_key = cache_key or name
        self.depends_on = depends_on or []
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @abstractmethod
    async def process_ws(
        self, ws: "WorkingSet", datastore: DataStore, **kwargs
    ) -> tuple["WorkingSet", StageResult]:
        """Process data using WorkingSet (DuckDB-native).

        All stages must implement this method.  Stages that need actual
        sequence data (e.g. ClusteringStage) should call
        ``datastore.materialize_working_set(ws)`` internally — that is
        *the stage's* responsibility, not the pipeline's.

        Args:
            ws: Input WorkingSet (set of sequence IDs).
            datastore: DataStore for reading/writing data.
            **kwargs: Additional arguments (e.g. ``run_id``).

        Returns:
            Tuple of (output WorkingSet, StageResult).
        """
        pass

    def to_spec(self) -> dict:
        """Return a serializable dict describing this stage.

        Used by BasePipeline.run_async() to persist pipeline definitions to DuckDB
        (enabling DataPipeline.from_db() recovery after kernel death).

        Subclasses must override this. Raises NotImplementedError by default.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement to_spec(). "
            "Implement to_spec() to enable pipeline definition saving and from_db() recovery."
        )

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Legacy DataFrame interface — used by streaming mode and GenerationStage.

        Subclasses may override this for backward compatibility or for cases
        where a DataFrame is the natural input (e.g. generation with an empty df).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement process()"
        )

    def __repr__(self):
        deps = f", depends_on={self.depends_on}" if self.depends_on else ""
        return f"{self.__class__.__name__}('{self.name}'{deps})"


@dataclass
class PipelineMetadata:
    """Metadata for a pipeline run — lets users retrieve and reuse cached results.

    Attributes:
        pipeline_id: Unique identifier for the pipeline's cache directory.
        cache_dir: Path to the ``.biolm/pipelines/<id>`` cache directory.
        db_path: Path to the DuckDB database file inside the cache directory.
        run_id: The run ID for this execution (there can be multiple runs
            sharing the same cache).

    Example::

        pipeline = DataPipeline(sequences=[...])
        pipeline.run()
        meta = pipeline.metadata
        print(meta.pipeline_id)   # "20260302_143022_a1b2c3d4"
        print(meta.cache_dir)     # ".biolm/pipelines/20260302_143022_a1b2c3d4"

        # Later — reuse the same cache:
        pipeline2 = DataPipeline(
            sequences=new_seqs,
            datastore=meta.db_path,   # or str(meta.cache_dir)
            resume=True,
        )
    """

    pipeline_id: str
    cache_dir: Path
    db_path: Path
    run_id: str


# Default cache root — lives alongside other dotfiles in the working directory
_BIOLM_CACHE_ROOT = Path(".biolm") / "pipelines"


class BasePipeline(ABC):
    """
    Base class for all pipeline types.

    Provides:
    - Stage management and dependency resolution
    - Async execution with progress tracking
    - Caching and resumability
    - Export and visualization

    When no ``datastore`` is provided, the pipeline automatically creates a
    DuckDB cache under ``.biolm/pipelines/<pipeline_id>/``.  The
    ``pipeline_id`` (and full cache path) is exposed via :attr:`metadata` so
    users can reconnect to the same cache in later sessions.

    Args:
        datastore: DataStore instance or path to a DuckDB file.  Required.
        run_id: Unique run identifier (auto-generated if not provided).
        output_dir: Directory for CSV/Parquet exports (default ``pipeline_outputs``).
        resume: Whether to resume from a previous run.
        verbose: Enable verbose output.
    """

    def __init__(
        self,
        datastore: Union[DataStore, str, Path],
        run_id: Optional[str] = None,
        output_dir: Union[str, Path] = "pipeline_outputs",
        resume: bool = False,
        verbose: bool = True,
        input_schema: Optional[InputSchema] = None,
    ):
        self.run_id = run_id or self._generate_run_id()

        # Setup datastore
        if isinstance(datastore, DataStore):
            self.datastore = datastore
            self._pipeline_id = self.run_id
            self._cache_dir = Path(datastore.db_path).parent
        elif isinstance(datastore, (str, Path)):
            self.datastore = DataStore(str(datastore))
            self._pipeline_id = self.run_id
            self._cache_dir = Path(datastore).parent
        else:
            raise TypeError(
                f"datastore must be a DuckDBDataStore instance or a path to a "
                f"DuckDB file, got {type(datastore).__name__}. Example: "
                f"DataPipeline(sequences=..., datastore='my_pipeline.duckdb')"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.verbose = verbose
        self.input_schema = input_schema

        # Pipeline context for inter-stage communication
        self.context = PipelineContext(self.datastore, self.run_id)

        # Stage management
        self.stages: list[Stage] = []
        self.stage_results: dict[str, StageResult] = {}
        # WorkingSet is now primary inter-stage transport
        self._working_sets: dict[str, WorkingSet] = {}
        # Legacy: kept for backward compat (populated lazily by get_final_data)
        self._stage_data: dict[str, pd.DataFrame] = {}

        # Pipeline state
        self.pipeline_type = self.__class__.__name__
        self.status = "initialized"
        self.start_time = None
        self.end_time = None

    @property
    def metadata(self) -> PipelineMetadata:
        """Return metadata for reconnecting to this pipeline's cache later."""
        return PipelineMetadata(
            pipeline_id=self._pipeline_id,
            cache_dir=self._cache_dir,
            db_path=Path(self.datastore.db_path),
            run_id=self.run_id,
        )

    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"

    def add_stage(self, stage: Stage):
        """Add a stage to the pipeline."""
        # Check for duplicate stage name
        existing_names = {s.name for s in self.stages}
        if stage.name in existing_names:
            raise ValueError(
                f"Duplicate stage name '{stage.name}'. "
                "Use stage_name= to provide a unique name."
            )

        # Check for column name collision.
        # Two stages may share output columns only if they have the exact same
        # (model_name, action) pair — meaning they are the same prediction and
        # one will just hit the cache.  Different model OR different action with
        # the same column name would silently overwrite DuckDB storage.
        stage_columns = {r.column for r in getattr(stage, "_resolved", [])}
        stage_key = (getattr(stage, "model_name", None), getattr(stage, "action", None))
        if stage_columns:
            for s in self.stages:
                s_columns = {r.column for r in getattr(s, "_resolved", [])}
                s_key = (getattr(s, "model_name", None), getattr(s, "action", None))
                overlap = stage_columns & s_columns
                if overlap and s_key != stage_key:
                    raise ValueError(
                        f"Column(s) {overlap} already used by stage '{s.name}' "
                        f"(model='{s_key[0]}', action='{s_key[1]}'). "
                        f"New stage '{stage.name}' uses model='{stage_key[0]}', "
                        f"action='{stage_key[1]}'. Use different column names via "
                        "the 'columns' parameter to avoid collision."
                    )

        # Cross-pipeline column registry check: ensure no column conflicts with
        # data stored in this DuckDB from a different (model, action) pair.
        if self.datastore and stage_columns:
            for r in getattr(stage, "_resolved", []):
                existing = self.datastore.get_column_registry_entry(r.column)
                if existing and (existing["model_name"], existing["action"]) != (
                    getattr(stage, "model_name", None),
                    getattr(stage, "action", None),
                ):
                    raise ValueError(
                        f"Column '{r.column}' was previously used by "
                        f"model='{existing['model_name']}', action='{existing['action']}' "
                        f"(definition '{existing['definition_id']}'). "
                        "Using a different (model, action) with the same column name would "
                        "overwrite existing predictions in DuckDB. "
                        "Use a different column name via the 'columns' parameter."
                    )

        # Validate dependencies
        for dep in stage.depends_on:
            if dep not in existing_names:
                raise ValueError(
                    f"Stage '{stage.name}' depends on '{dep}' which hasn't been added yet"
                )

        self.stages.append(stage)
        if self.verbose:
            print(f"Added stage: {stage}")

    def _save_definition_and_register_columns(self):
        """Serialize stages, save definition to DuckDB, register output columns."""
        if self.datastore is None:
            return
        from biolmai.pipeline.pipeline_def import _pipeline_def_hash

        stages_specs = []
        for s in self.stages:
            try:
                stages_specs.append(s.to_spec())
            except NotImplementedError:
                # Stage doesn't support serialization — store minimal info
                stages_specs.append({"type": s.__class__.__name__, "name": s.name})

        input_cols = self.input_schema.columns if self.input_schema else None
        def_id = _pipeline_def_hash(self.pipeline_type, input_cols, stages_specs)
        self.datastore.save_pipeline_definition(
            def_id,
            self.pipeline_type,
            json.dumps(input_cols) if input_cols is not None else None,
            json.dumps(stages_specs),
        )
        for s in self.stages:
            for r in getattr(s, "_resolved", []):
                self.datastore.register_column(
                    r.column,
                    getattr(s, "model_name", ""),
                    getattr(s, "action", ""),
                    def_id,
                    s.name,
                )
        self.datastore.conn.execute(
            "UPDATE pipeline_runs SET definition_id = ? WHERE run_id = ?",
            [def_id, self.run_id],
        )

    @classmethod
    def from_db(
        cls,
        db_path: Union[str, "Path"],
        definition_id: Optional[str] = None,
        run_id: Optional[str] = None,
        verbose: bool = True,
    ) -> "BasePipeline":
        """Reconstruct a pipeline from an existing DuckDB database.

        Useful for recovering after a kernel death without re-running
        already-completed stages.

        Args:
            db_path: Path to the DuckDB database file.
            definition_id: Specific definition to load (``None`` = latest).
            run_id: Run ID for the reconstructed pipeline (``None`` = generate new).
            verbose: Enable verbose output.

        Returns:
            Reconstructed :class:`BasePipeline` subclass instance.

        Example::

            pipeline = DataPipeline.from_db("my_pipeline.duckdb")
            pipeline.run(resume=True)
        """
        from biolmai.pipeline.pipeline_def import pipeline_from_definition

        ds = DataStore(str(db_path))
        defn = ds.load_pipeline_definition(definition_id)
        if defn is None:
            raise ValueError(
                f"No pipeline definition found in '{db_path}'. "
                "Run a pipeline with this datastore first to persist a definition."
            )
        return pipeline_from_definition(defn, datastore=ds, run_id=run_id, verbose=verbose)

    def _resolve_dependencies(self) -> list[list[Stage]]:
        """
        Resolve stage dependencies and return execution order.

        Returns:
            List of stage groups, where stages in each group can run in parallel
        """
        # Build dependency graph
        stage_map = {s.name: s for s in self.stages}

        # Topological sort with level detection
        in_degree = {s.name: len(s.depends_on) for s in self.stages}
        levels = []

        while in_degree:
            # Find all stages with no dependencies
            current_level = [
                stage_map[name] for name, degree in in_degree.items() if degree == 0
            ]

            if not current_level:
                remaining = list(in_degree.keys())
                raise ValueError(
                    f"Circular dependency detected among stages: {remaining}"
                )

            levels.append(current_level)

            # Remove current level from graph
            for stage in current_level:
                del in_degree[stage.name]

            # Decrease in-degree for dependent stages
            for stage in current_level:
                for other_stage in self.stages:
                    if (
                        stage.name in other_stage.depends_on
                        and other_stage.name in in_degree
                    ):
                        in_degree[other_stage.name] -= 1

        return levels

    def _reload_stage_working_set(
        self,
        stage: Stage,
        ws_input: WorkingSet,
    ) -> Optional[WorkingSet]:
        """Reconstruct a stage's WorkingSet from the datastore (for resume).

        Returns the reloaded WorkingSet, or None if it cannot be reconstructed
        (caller should then re-run the stage normally).
        """
        if self.datastore is None:
            return None
        try:
            from biolmai.pipeline.data import (
                ClusteringStage,
                FilterStage,
                PredictionStage,
            )
            from biolmai.pipeline.generative import GenerationStage

            if isinstance(stage, PredictionStage):
                # Encode stages store to the embeddings table, not predictions.
                # We can't reconstruct their WS from predictions — force re-run.
                if getattr(stage, "action", None) == "encode":
                    return None

                # For predict/score: IDs that have ALL resolved columns in DuckDB.
                resolved = getattr(stage, "_resolved", [])
                if not resolved:
                    return None
                candidate_ids = set(ws_input.sequence_ids)
                for r in resolved:
                    ids_with_col = self.datastore.get_sequence_ids_with_prediction(
                        list(candidate_ids),
                        r.column,
                        stage.model_name,
                    )
                    candidate_ids &= set(ids_with_col)
                    if not candidate_ids:
                        return None
                return WorkingSet.from_ids(candidate_ids)

            elif isinstance(stage, FilterStage):
                passed_ids = self.datastore.get_filter_results(self.run_id, stage.name)
                if not passed_ids:
                    return None
                return WorkingSet.from_ids(passed_ids)

            elif isinstance(stage, ClusteringStage):
                # Clustering doesn't filter — return same set
                assignments = self.datastore.get_pipeline_metadata(
                    f"clustering_{stage.name}_assignments"
                )
                if assignments is None:
                    return None
                return ws_input  # same IDs, clustering adds columns at materialize time

            elif isinstance(stage, GenerationStage):
                # Generation creates new sequences — get all from datastore
                df = self.datastore.export_to_dataframe(
                    include_sequences=True,
                    include_predictions=False,
                    include_generation_metadata=True,
                )
                if df.empty or "sequence_id" not in df.columns:
                    return None
                return WorkingSet.from_ids(df["sequence_id"].tolist())

        except Exception as exc:
            _logger.warning(
                "Could not reload stage '%s' from datastore: %s", stage.name, exc
            )

        return None

    # Legacy compat: kept for tests that reference _reload_stage_output
    def _reload_stage_output(
        self,
        stage: Stage,
        df_input: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Reconstruct a stage's output DataFrame from the datastore (for resume)."""
        if self.datastore is None or "sequence_id" not in df_input.columns:
            return None
        ws_input = WorkingSet.from_ids(df_input["sequence_id"].tolist())
        ws_out = self._reload_stage_working_set(stage, ws_input)
        if ws_out is None:
            return None
        return self.datastore.materialize_working_set(ws_out)

    async def _execute_stage_ws(
        self, stage: Stage, ws_input: WorkingSet
    ) -> tuple[WorkingSet, StageResult]:
        """Execute a single stage using WorkingSet transport."""
        start_time = time.time()

        # Check if stage is already complete (resumability)
        stage_id = f"{self.run_id}_{stage.name}"
        if (
            self.resume
            and self.datastore
            and self.datastore.is_stage_complete(stage_id)
        ):
            if self.verbose:
                print(f"\n✓ Stage '{stage.name}' already complete — reloading from DB")

            ws_resumed = self._reload_stage_working_set(stage, ws_input)
            if ws_resumed is not None:
                if self.verbose:
                    print(f"  Reloaded {len(ws_resumed):,} sequences")
                return ws_resumed, StageResult(
                    stage_name=stage.name,
                    input_count=len(ws_input),
                    output_count=len(ws_resumed),
                    elapsed_time=0.0,
                    metadata={"resumed": True},
                )
            else:
                if self.verbose:
                    print(f"  Cannot reload from DB — re-running stage '{stage.name}'")

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Stage: {stage.name}")
            print(f"Input: {len(ws_input):,} sequences")
            if stage.depends_on:
                print(f"Depends on: {', '.join(stage.depends_on)}")

        # Execute stage — all stages implement process_ws()
        ws_out, result = await stage.process_ws(
            ws_input, self.datastore, run_id=self.run_id, context=self.context
        )

        result.elapsed_time = time.time() - start_time

        # Mark stage complete
        if self.datastore:
            self.datastore.mark_stage_complete(
                stage_id=stage_id,
                run_id=self.run_id,
                stage_name=stage.name,
                input_count=result.input_count,
                output_count=result.output_count,
                status="completed",
            )

        if self.verbose:
            print(f"\n{result}")
            print(f"{'='*60}")

        return ws_out, result

    async def run_async(
        self, enable_streaming: bool = True, **kwargs
    ) -> dict[str, StageResult]:
        """
        Run the pipeline asynchronously.

        Args:
            enable_streaming: Stream prediction results through per-sequence
                filters for better parallelism and lower latency (default True).

        Returns:
            Dict mapping stage names to StageResults
        """
        self.start_time = time.time()
        self.status = "running"

        # Create pipeline run record
        config = self._get_config()
        self.datastore.create_pipeline_run(
            run_id=self.run_id,
            pipeline_type=self.pipeline_type,
            config=config,
            status="running",
        )

        try:
            # Save pipeline definition (content-hash dedup) + register output columns
            self._save_definition_and_register_columns()

            # Get initial data — returns WorkingSet
            ws_current = await self._get_initial_data_ws(**kwargs)

            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Pipeline: {self.pipeline_type}")
                print(f"# Run ID: {self.run_id}")
                print(f"# Initial sequences: {len(ws_current):,}")
                if enable_streaming:
                    print("# Streaming: ENABLED")
                print(f"{'#'*60}")

            # Resolve dependencies and get execution order
            stage_levels = self._resolve_dependencies()

            if self.verbose:
                print(f"\nExecution plan: {len(stage_levels)} level(s)")
                for i, level in enumerate(stage_levels):
                    stage_names = [s.name for s in level]
                    parallel_str = " (parallel)" if len(level) > 1 else ""
                    print(f"  Level {i+1}: {', '.join(stage_names)}{parallel_str}")

            # Execute stages level by level
            processed_stages = set()  # Track which stages we've already processed

            for level_idx, level_stages in enumerate(stage_levels):
                if len(level_stages) == 1:
                    # Single stage in level
                    stage = level_stages[0]

                    # Skip if already processed via streaming
                    if stage.name in processed_stages:
                        continue

                    # Check if we can stream through this stage
                    next_stage = self._get_next_stage(stage, stage_levels, level_idx)
                    can_stream = (
                        enable_streaming
                        and next_stage is not None
                        and next_stage.name not in processed_stages
                        and hasattr(stage, "process_streaming")
                        and self._can_stream_to_next(stage, next_stage)
                    )

                    if can_stream:
                        # STREAMING: uses legacy DataFrame path
                        df_current = self.datastore.materialize_working_set(ws_current)
                        df_out = await self._execute_stage_streaming(
                            stage, next_stage, df_current
                        )
                        self._stage_data[stage.name] = df_out
                        self._stage_data[next_stage.name] = df_out
                        if "sequence_id" in df_out.columns:
                            ws_current = WorkingSet.from_ids(
                                df_out["sequence_id"].tolist()
                            )
                        self._working_sets[stage.name] = ws_current
                        self._working_sets[next_stage.name] = ws_current
                        processed_stages.add(stage.name)
                        processed_stages.add(next_stage.name)
                    else:
                        # BATCHING: WorkingSet path
                        ws_out, result = await self._execute_stage_ws(
                            stage, ws_current
                        )
                        self.stage_results[stage.name] = result
                        self._working_sets[stage.name] = ws_out
                        ws_current = ws_out
                        processed_stages.add(stage.name)
                else:
                    # Multiple stages in level - execute in parallel
                    if self.verbose:
                        print(f"\nExecuting {len(level_stages)} stages in parallel...")

                    tasks = [
                        self._execute_stage_ws(stage, ws_current)
                        for stage in level_stages
                    ]
                    results = await asyncio.gather(*tasks)

                    # Parallel merge = set intersection (was: DataFrame merge)
                    all_working_sets = []
                    for stage, (ws_out, result) in zip(level_stages, results):
                        self.stage_results[stage.name] = result
                        self._working_sets[stage.name] = ws_out
                        all_working_sets.append(ws_out)

                    # Intersection of all parallel stage outputs
                    if all_working_sets:
                        ws_merged = all_working_sets[0]
                        for ws in all_working_sets[1:]:
                            ws_merged = ws_merged.intersect(ws)
                        ws_current = ws_merged
                    else:
                        ws_current = WorkingSet(frozenset())

                    # Update the last stage's working set to the merged result
                    if level_stages:
                        self._working_sets[level_stages[-1].name] = ws_current

            self.status = "completed"
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, "completed")

            if self.verbose:
                total_time = self.end_time - self.start_time
                print(f"\n{'#'*60}")
                print(f"# Pipeline completed in {total_time:.1f}s")
                print(f"# Final sequences: {len(ws_current):,}")
                print(f"{'#'*60}\n")

            return self.stage_results

        except Exception:
            self.status = "failed"
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, "failed")
            raise

    async def _get_initial_data_ws(self, **kwargs) -> WorkingSet:
        """Get initial WorkingSet for the pipeline.

        Default implementation calls the legacy _get_initial_data() and converts.
        Subclasses can override for a more efficient direct path.
        """
        df = await self._get_initial_data(**kwargs)
        if "sequence_id" in df.columns:
            return WorkingSet.from_ids(df["sequence_id"].tolist())
        return WorkingSet(frozenset())

    def _get_next_stage(
        self, current_stage: Stage, stage_levels: list[list[Stage]], current_level: int
    ) -> Optional[Stage]:
        """Get the next stage after current_stage, if any."""
        if current_level + 1 >= len(stage_levels):
            return None

        next_level = stage_levels[current_level + 1]
        if len(next_level) == 1:
            return next_level[0]
        return None  # Can't stream to multiple parallel stages

    def _can_stream_to_next(self, current_stage: Stage, next_stage: Stage) -> bool:
        """Check if current stage can stream to next stage."""
        from biolmai.pipeline.data import FilterStage

        # Can stream if next stage is a filter that doesn't require complete data
        if isinstance(next_stage, FilterStage):
            return not next_stage.requires_complete_data

        # Can also stream to another prediction stage
        from biolmai.pipeline.data import PredictionStage

        if isinstance(next_stage, PredictionStage):
            return True

        return False

    async def _execute_stage_streaming(
        self, stage: Stage, next_stage: Stage, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Execute stage in streaming mode, passing results to next_stage incrementally.

        Returns:
            Final output DataFrame after both stages
        """
        from biolmai.pipeline.data import FilterStage

        if self.verbose:
            print(f"\n[Stage: {stage.name}] (streaming to {next_stage.name})")

        # Collect output chunks
        output_chunks = []
        processed_count = 0
        filtered_count = 0

        # Stream through both stages
        async for chunk_df in stage.process_streaming(df, self.datastore):
            # Pass chunk through next stage immediately
            if isinstance(next_stage, FilterStage):
                # Filter the chunk
                start_chunk_count = len(chunk_df)
                filtered_chunk = next_stage.filter_func(chunk_df)
                filtered_count += start_chunk_count - len(filtered_chunk)

                if len(filtered_chunk) > 0:
                    output_chunks.append(filtered_chunk)
                    processed_count += len(filtered_chunk)

                    if self.verbose and processed_count % 100 == 0:
                        print(f"  Processed: {processed_count} sequences (streaming)")

        # Combine all chunks
        if output_chunks:
            df_out = pd.concat(output_chunks, ignore_index=True)
        else:
            df_out = pd.DataFrame(columns=df.columns)

        # Record results for both stages
        self.stage_results[stage.name] = StageResult(
            stage_name=stage.name,
            input_count=len(df),
            output_count=len(df),  # All sequences processed
            filtered_count=0,
        )

        self.stage_results[next_stage.name] = StageResult(
            stage_name=next_stage.name,
            input_count=len(df),
            output_count=len(df_out),
            filtered_count=filtered_count,
        )

        # Persist filter results so resume can reload the filter stage output
        if (
            self.datastore
            and isinstance(next_stage, FilterStage)
            and "sequence_id" in df_out.columns
        ):
            self.datastore.save_filter_results(
                self.run_id, next_stage.name, df_out["sequence_id"].tolist()
            )

        # Bug #1 fix: mark both stages complete so resume logic works correctly
        if self.datastore:
            stage_id = f"{self.run_id}_{stage.name}"
            self.datastore.mark_stage_complete(
                run_id=self.run_id,
                stage_name=stage.name,
                stage_id=stage_id,
                input_count=len(df),
                output_count=len(df),
                status="completed",
            )
            next_stage_id = f"{self.run_id}_{next_stage.name}"
            self.datastore.mark_stage_complete(
                run_id=self.run_id,
                stage_name=next_stage.name,
                stage_id=next_stage_id,
                input_count=len(df),
                output_count=len(df_out),
                status="completed",
            )

        if self.verbose:
            print(f"  {stage.name}: processed {len(df)} sequences")
            print(
                f"  {next_stage.name}: {len(df_out)} passed filter (filtered {filtered_count})"
            )

        return df_out

    def run(self, enable_streaming: bool = True, **kwargs) -> dict[str, StageResult]:
        """
        Run the pipeline synchronously.

        This is a convenience wrapper around run_async().
        """
        return asyncio.run(self.run_async(enable_streaming=enable_streaming, **kwargs))

    @abstractmethod
    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """
        Get initial DataFrame for the pipeline.

        Must return DataFrame with at least 'sequence' column.
        Should add 'sequence_id' column by inserting into datastore.
        """
        pass

    def _get_config(self) -> dict:
        """Get pipeline configuration for serialization."""
        return {
            "pipeline_type": self.pipeline_type,
            "run_id": self.run_id,
            "stages": [
                {
                    "name": s.name,
                    "type": s.__class__.__name__,
                    "depends_on": s.depends_on,
                }
                for s in self.stages
            ],
        }

    def get_final_data(self) -> pd.DataFrame:
        """Get the final output DataFrame.

        Materializes from the last stage's WorkingSet via DuckDB.
        Falls back to legacy _stage_data if WorkingSet is not available.
        """
        if not self._working_sets and not self._stage_data:
            raise RuntimeError("Pipeline has not been run yet")

        last_stage_name = self.stages[-1].name

        # Primary path: materialize from WorkingSet
        if last_stage_name in self._working_sets:
            ws = self._working_sets[last_stage_name]
            # Check if we already have a cached df (from legacy stage)
            if last_stage_name in self._stage_data:
                return self._stage_data[last_stage_name]
            return self.datastore.materialize_working_set(ws)

        # Legacy fallback
        return self._stage_data.get(last_stage_name, pd.DataFrame())

    def export_to_csv(self, output_path: Optional[Union[str, Path]] = None):
        """Export final results to CSV."""
        if output_path is None:
            output_path = self.output_dir / f"{self.run_id}_final.csv"

        df = self.get_final_data()
        df.to_csv(output_path, index=False)

        if self.verbose:
            print(f"Exported {len(df)} sequences to {output_path}")

    def summary(self) -> pd.DataFrame:
        """Get pipeline summary statistics."""
        if not self.stage_results:
            print("Pipeline has not been run yet")
            return pd.DataFrame()

        summary_data = []
        for _stage_name, result in self.stage_results.items():
            summary_data.append(
                {
                    "Stage": result.stage_name,
                    "Input": result.input_count,
                    "Output": result.output_count,
                    "Filtered": result.filtered_count,
                    "Cached": result.cached_count,
                    "Computed": result.computed_count,
                    "Time (s)": f"{result.elapsed_time:.1f}",
                }
            )

        return pd.DataFrame(summary_data)

    def __repr__(self):
        return (
            f"{self.pipeline_type}("
            f"run_id='{self.run_id}', "
            f"stages={len(self.stages)}, "
            f"status='{self.status}')"
        )
