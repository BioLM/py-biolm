"""
Data-driven pipeline implementations.

DataPipeline: Load sequences from files/lists and run predictions
SingleStepPipeline: Simplified single-step prediction pipeline
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from biolmai.client import BioLMApiClient  # Use async client directly
from biolmai.pipeline.base import BasePipeline, InputSchema, Stage, StageResult, WorkingSet
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolmai.pipeline.filters import BaseFilter


@dataclass
class ExtractionSpec:
    """Specification for extracting a value from an API response (with reduction).

    Use when you need to apply a reduction (mean, max, min, sum) to an
    array-valued response key. For simple scalar extractions, pass a plain
    string to ``extractions`` instead.

    Args:
        response_key: Key in API response dict, e.g. "plddt"
        reduction: Optional reduction for array values: "mean", "max", "min", "sum"
    """

    response_key: str
    reduction: Optional[str] = None


@dataclass
class _ResolvedExtraction:
    """Internal: fully resolved extraction with output column name.

    Created by PredictionStage.__init__ from the user-facing ``extractions``
    and ``columns`` parameters.

    Attributes:
        response_key: Key to read from the API response dict.
        column: Output column name (DuckDB ``prediction_type`` + DataFrame column).
        reduction: Optional reduction for array values.
    """

    response_key: str
    column: str
    reduction: Optional[str] = None


def _resolve_extractions(
    extractions: Union[str, list[Union[str, ExtractionSpec]]],
    columns: Optional[Union[str, dict[str, str]]] = None,
) -> list[_ResolvedExtraction]:
    """Normalize user-facing extractions + columns into _ResolvedExtraction list.

    Examples::

        _resolve_extractions("prediction", "tm")
        → [_ResolvedExtraction("prediction", "tm")]

        _resolve_extractions("prediction")
        → [_ResolvedExtraction("prediction", "prediction")]

        _resolve_extractions(["mean_plddt", "ptm"], {"mean_plddt": "plddt"})
        → [_ResolvedExtraction("mean_plddt", "plddt"), _ResolvedExtraction("ptm", "ptm")]

        _resolve_extractions([ExtractionSpec("plddt", reduction="mean")], {"plddt": "plddt_mean"})
        → [_ResolvedExtraction("plddt", "plddt_mean", "mean")]
    """
    # Normalize extractions to list of (response_key, reduction)
    specs: list[tuple[str, Optional[str]]] = []
    if isinstance(extractions, str):
        specs = [(extractions, None)]
    elif isinstance(extractions, list):
        for item in extractions:
            if isinstance(item, str):
                specs.append((item, None))
            elif isinstance(item, ExtractionSpec):
                specs.append((item.response_key, item.reduction))
            else:
                raise TypeError(
                    f"extractions list items must be str or ExtractionSpec, got {type(item)}"
                )
    else:
        raise TypeError(
            f"extractions must be str or list, got {type(extractions)}"
        )

    # Build column mapping
    col_map: dict[str, str] = {}
    if columns is None:
        pass  # identity: response_key = column name
    elif isinstance(columns, str):
        if len(specs) != 1:
            raise ValueError(
                f"columns='{columns}' (str) can only be used with a single extraction, "
                f"got {len(specs)}. Use a dict for multiple extractions."
            )
        col_map[specs[0][0]] = columns
    elif isinstance(columns, dict):
        col_map = dict(columns)
    else:
        raise TypeError(f"columns must be str, dict, or None, got {type(columns)}")

    # Resolve
    resolved = []
    for response_key, reduction in specs:
        column = col_map.get(response_key, response_key)
        resolved.append(_ResolvedExtraction(response_key, column, reduction))

    return resolved


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

# Return type for embedding extractors: list of (array, optional_layer_number)
EmbeddingResult = list[tuple[np.ndarray, Optional[int]]]


@dataclass
class EmbeddingSpec:
    """Declarative specification for extracting embeddings from API responses.

    Covers common response formats without writing a custom function.

    Args:
        key: Response dict key containing the embedding data (e.g.
            ``"embedding"``, ``"seqcoding"``, ``"embeddings"``).
        layer: Which layer to extract when the response contains multiple
            layers (list of ``{layer: int, embedding: [...]}`` dicts).
            ``None`` stores all layers; an ``int`` stores only that layer.
        reduction: Reduce per-token 2-D embeddings to a single vector:
            ``"mean"``, ``"first"``, ``"last"``, ``"sum"``.
            ``None`` stores the full array as-is.

    Examples::

        # ablang2 returns {"seqcoding": [float, ...]}
        EmbeddingSpec(key="seqcoding")

        # esm2-8m returns {"embeddings": [{embedding: [...], layer: 33}]}
        # Store only layer 33:
        EmbeddingSpec(key="embeddings", layer=33)

        # Per-residue → mean-pool:
        EmbeddingSpec(key="embedding", reduction="mean")
    """

    key: str
    layer: Optional[int] = None
    reduction: Optional[str] = None

    def __call__(self, result: dict) -> EmbeddingResult:
        """Extract embeddings from an API response dict."""
        val = result.get(self.key)
        if val is None:
            return []

        # Case 1: flat list of floats or a nested numeric array
        if isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            if arr.size == 0:
                return []

            # Check if it's a list of dicts (multi-layer format)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return self._extract_layers(val)

            # Apply reduction if requested (e.g. per-token 2-D → 1-D)
            arr = self._apply_reduction(arr)
            return [(arr, None)]

        return []

    def _extract_layers(self, items: list[dict]) -> EmbeddingResult:
        """Handle list-of-dicts format: [{layer: 0, embedding: [...]}, ...]."""
        results: EmbeddingResult = []
        for item in items:
            layer_num = item.get("layer")
            emb_data = item.get("embedding")
            if emb_data is None:
                continue
            if self.layer is not None and layer_num != self.layer:
                continue
            arr = np.array(emb_data)
            if arr.size > 0:
                arr = self._apply_reduction(arr)
                results.append((arr, layer_num))
        return results

    def _apply_reduction(self, arr: np.ndarray) -> np.ndarray:
        """Reduce a 2-D array to 1-D if a reduction is set."""
        if self.reduction is None or arr.ndim < 2:
            return arr
        if self.reduction == "mean":
            return arr.mean(axis=0)
        elif self.reduction == "first":
            return arr[0]
        elif self.reduction == "last":
            return arr[-1]
        elif self.reduction == "sum":
            return arr.sum(axis=0)
        return arr


class PredictionStage(Stage):
    """
    Generic prediction stage using BioLM API.

    Args:
        name: Stage name
        model_name: BioLM model name (e.g., 'esmfold', 'esm2', 'temberture-regression')
        action: API action ('predict', 'encode', 'score')
        prediction_type: Type of prediction for caching (e.g., 'structure', 'stability', 'embedding')
        params: Optional parameters for the API call
        batch_size: Number of sequences per pipeline batch (default 32).
            Each pipeline batch becomes one SDK call, which the SDK may
            further split by the model's ``maxItems`` schema limit.
        max_concurrent: Maximum pipeline batches in flight at once (default 5).
            Controls how many batches are dispatched concurrently at the
            pipeline level.  Higher values keep the API more saturated but
            use more memory (up to ``max_concurrent * batch_size`` items
            plus their responses in memory at once).
        max_connections: Maximum concurrent HTTP connections to the API
            (default 10).  This is the SDK-level semaphore that throttles
            the actual HTTP requests.  Each pipeline batch may be split
            into multiple sub-requests by the model's ``maxItems`` limit;
            ``max_connections`` caps how many of those sub-requests run
            simultaneously across all in-flight batches.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        action: str = "predict",
        params: Optional[dict] = None,
        batch_size: int = 32,
        max_concurrent: int = 5,
        max_connections: int = 10,
        item_columns: Optional[dict[str, str]] = None,
        extractions: Optional[Union[str, list[Union[str, ExtractionSpec]]]] = None,
        columns: Optional[Union[str, dict[str, str]]] = None,
        embedding_extractor: Optional[Union[EmbeddingSpec, callable]] = None,
        **kwargs,
    ):
        # Extract skip_on_error before passing to parent
        skip_on_error = kwargs.pop("skip_on_error", False)

        self.action = action
        self.params = params or {}
        self.batch_size = batch_size
        self.skip_on_error = skip_on_error
        # SDK-level HTTP connection semaphore — separate from pipeline flight limit
        self._connection_semaphore = asyncio.Semaphore(max_connections)
        # item_columns: maps API field name → DataFrame column name.
        # E.g. {'H': 'heavy_chain', 'L': 'light_chain'} for abodybuilder3.
        # When None, defaults to {'sequence': 'sequence'}.
        self.item_columns = item_columns
        # Reuse API client across calls for connection pooling
        self._api_client = None

        # Embedding extraction: user-defined function or EmbeddingSpec.
        self._embedding_extractor = embedding_extractor

        # --- Resolve extractions + columns into list[_ResolvedExtraction] ---
        self._resolved: list[_ResolvedExtraction] = []

        if action in ("predict", "score"):
            if extractions is None:
                raise ValueError(
                    f"PredictionStage '{name}': `extractions` is required. "
                    f"Set extractions='response_key' or "
                    f"extractions=[ExtractionSpec('key', reduction='mean')]. "
                    f"Check your model's API response to find the correct key."
                )
            self._resolved = _resolve_extractions(extractions, columns)

        if action == "encode" and self._embedding_extractor is None:
            raise ValueError(
                f"PredictionStage '{name}': `embedding_extractor` is required. "
                f"Set embedding_extractor=EmbeddingSpec('response_key') or "
                f"pass a callable. Check your model's API response for the "
                f"embedding key (e.g. 'embedding', 'seqcoding', 'embeddings')."
            )

        # prediction_type: first column name (used for cache checks, stage naming)
        self.prediction_type = (
            self._resolved[0].column if self._resolved else f"{model_name}_{action}"
        )

        # Cache key: auto-derived from model + action + sorted response keys
        response_keys = sorted({r.response_key for r in self._resolved})
        cache_key = (
            f"{model_name}::{action}::{','.join(response_keys)}"
            if response_keys
            else f"{model_name}_{action}"
        )

        super().__init__(
            name=name,
            cache_key=cache_key,
            model_name=model_name,
            max_concurrent=max_concurrent,
            **kwargs,
        )

    async def process_streaming(self, df: pd.DataFrame, datastore: DataStore, **kwargs):
        """
        Process sequences and yield results as batches complete (streaming).

        Yields DataFrames as API batches complete instead of waiting for all results.
        This allows downstream stages to start processing immediately.
        """

        start_count = len(df)

        # Ensure sequence_ids are present (normally added by _get_initial_data)
        if "sequence_id" not in df.columns:
            df = df.copy()
            df["sequence_id"] = datastore.add_sequences_batch(df["sequence"].tolist())

        # Vectorized cache check: single anti-join, not N individual has_prediction() calls
        uncached_ids = datastore.get_uncached_sequence_ids(
            df["sequence_id"].tolist(), self.prediction_type, self.model_name
        )
        uncached_mask = df["sequence_id"].isin(uncached_ids)
        df_uncached = df[uncached_mask].copy()
        cached_count = start_count - len(df_uncached)

        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)} (streaming)")

        # Yield cached results first using bulk fetch (single JOIN, not N queries)
        if cached_count > 0:
            df_cached = df[~uncached_mask].copy()
            merge_specs = self._resolved
            cached_seq_ids = df_cached["sequence_id"].tolist()
            for spec in merge_specs:
                pred_df = datastore.get_predictions_bulk(
                    cached_seq_ids, spec.column, self.model_name
                )
                if not pred_df.empty:
                    val_map = dict(zip(pred_df["sequence_id"], pred_df["value"]))
                    df_cached[spec.column] = df_cached["sequence_id"].map(val_map)
            yield df_cached

        if len(df_uncached) == 0:
            return

        # Create or reuse async API client
        if self._api_client is None:
            self._api_client = BioLMApiClient(
                self.model_name, semaphore=self._connection_semaphore, retry_error_batches=True
            )
        api = self._api_client

        # Batch sequences for API calls
        batch_size = self.batch_size

        # Create tasks for all batches and start them immediately
        pending_tasks = {}  # task -> (batch_df, batch_indices)

        for i in range(0, len(df_uncached), batch_size):
            batch_df = df_uncached.iloc[i : i + batch_size]
            batch_indices = batch_df.index

            if self.item_columns:
                items = [
                    {
                        api_field: row[col]
                        for api_field, col in self.item_columns.items()
                    }
                    for _, row in batch_df.iterrows()
                ]
            else:
                items = [{"sequence": seq} for seq in batch_df["sequence"].tolist()]

            # Create and start task immediately
            if self.action == "encode":
                task = asyncio.create_task(api.encode(items=items, params=self.params))
            elif self.action == "score":
                task = asyncio.create_task(api.score(items=items, params=self.params))
            else:
                task = asyncio.create_task(api.predict(items=items, params=self.params))

            pending_tasks[task] = (batch_df, batch_indices)

        # Process batches as they complete (true streaming!)
        remaining_tasks = set(pending_tasks.keys())

        while remaining_tasks:
            # Wait for next batch to complete
            done, remaining_tasks = await asyncio.wait(
                remaining_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for completed_task in done:
                try:
                    results = await completed_task
                    pending_batch_df, batch_indices = pending_tasks[completed_task]
                    out_df = pending_batch_df.copy()

                    # Store results and add to DataFrame.
                    # Collect into batch_data and flush once (not per-row add_prediction).
                    batch_data = []
                    for (_, row), result in zip(pending_batch_df.iterrows(), results):
                        seq_id = int(row["sequence_id"])
                        idx = row.name

                        if self.action in ("predict", "score"):
                            if isinstance(result, dict):
                                for spec in self._resolved:
                                    val = self._extract_with_spec(result, spec)
                                    if val is not None:
                                        batch_data.append(
                                            {
                                                "sequence_id": seq_id,
                                                "prediction_type": spec.column,
                                                "model_name": self.model_name,
                                                "value": val,
                                                "metadata": {"params": self.params},
                                            }
                                        )
                                        out_df.at[idx, spec.column] = val

                        elif self.action == "encode":
                            self._store_embeddings(datastore, seq_id, result)

                    # Single batch insert per completed async task
                    if batch_data:
                        datastore.add_predictions_batch(batch_data)

                    # Yield batch immediately!
                    yield out_df

                except Exception as e:
                    if self.skip_on_error:
                        pending_batch_df, _ = pending_tasks[completed_task]
                        print(f"  Error processing batch (skipped): {e}")
                        failed_batch = [
                            {
                                "sequence_id": int(sid),
                                "prediction_type": self.prediction_type,
                                "model_name": self.model_name,
                                "value": None,
                                "metadata": {
                                    "status": "failed",
                                    "error": str(e),
                                    "params": self.params,
                                },
                            }
                            for sid in pending_batch_df["sequence_id"].tolist()
                        ]
                        datastore.add_predictions_batch(failed_batch)
                        # Don't yield failed batch - sequences are filtered out
                    else:
                        print(f"  Error processing batch: {e}")
                        raise

    @staticmethod
    def _extract_with_spec(
        result: dict, spec: Union[ExtractionSpec, _ResolvedExtraction]
    ) -> Optional[float]:
        """Extract a value from an API result using an extraction spec.

        Looks up spec.response_key in the result dict. If the value is a list/array
        and spec.reduction is set, applies the reduction (mean/max/min/sum).
        """
        val = result.get(spec.response_key)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, list) and val:
            # Array value — apply reduction
            try:
                arr = [float(x) for x in val if isinstance(x, (int, float))]
            except (TypeError, ValueError):
                return None
            if not arr:
                return None
            reduction = spec.reduction or "mean"
            if reduction == "mean":
                return float(np.mean(arr))
            elif reduction == "max":
                return float(np.max(arr))
            elif reduction == "min":
                return float(np.min(arr))
            elif reduction == "sum":
                return float(np.sum(arr))
            return float(np.mean(arr))
        return None

    def _store_embeddings(self, datastore: DataStore, seq_id: int, result: Any):
        """Store embedding(s) from a single API result dict into the datastore.

        Uses ``self._embedding_extractor`` (``EmbeddingSpec`` or callable)
        to extract embeddings from the API response.
        """
        if not isinstance(result, dict):
            return

        extracted = self._embedding_extractor(result)

        for embedding, layer in extracted:
            if embedding.size > 0:
                datastore.add_embedding(seq_id, self.model_name, embedding, layer=layer)

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Process sequences through prediction model (legacy DataFrame path).

        Uses the same bounded-concurrency batching as ``process_ws`` — see
        that method's docstring for the design rationale.

        Performance: single DuckDB anti-join for cache detection, bounded-
        concurrency API dispatch, single JOIN query to merge predictions back.
        """
        start_count = len(df)

        # Ensure sequence_ids are present (normally added by _get_initial_data)
        if "sequence_id" not in df.columns:
            df = df.copy()
            df["sequence_id"] = datastore.add_sequences_batch(df["sequence"].tolist())

        # --- Vectorized cache check: single anti-join, not N individual queries ---
        uncached_ids = datastore.get_uncached_sequence_ids(
            df["sequence_id"].tolist(), self.prediction_type, self.model_name
        )
        uncached_mask = df["sequence_id"].isin(uncached_ids)
        df_uncached = df[uncached_mask]
        cached_count = start_count - len(df_uncached)

        print(f"  Cached: {cached_count}/{start_count}")
        print(f"  To compute: {len(df_uncached)}")

        if len(df_uncached) > 0:
            print(f"  Calling {self.model_name}.{self.action}...")

            # Reuse client across calls — DO NOT shut down in finally
            if self._api_client is None:
                self._api_client = BioLMApiClient(
                    self.model_name, semaphore=self._connection_semaphore, retry_error_batches=True
                )
            api = self._api_client

            seq_ids = df_uncached["sequence_id"].tolist()
            seq_id_to_seq = dict(
                zip(df_uncached["sequence_id"], df_uncached["sequence"])
            )

            # Build items list
            if self.item_columns:
                all_items = [
                    {
                        api_field: row[col]
                        for api_field, col in self.item_columns.items()
                    }
                    for _, row in df_uncached.iterrows()
                ]
            else:
                all_items = [
                    {"sequence": seq} for seq in df_uncached["sequence"].tolist()
                ]

            # --- Bounded-concurrency dispatch (same pattern as process_ws) ---
            batch_size = self.batch_size
            n_batches = (len(all_items) + batch_size - 1) // batch_size
            flight_semaphore = asyncio.Semaphore(self.max_concurrent)

            async def _dispatch_batch(batch_idx):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(all_items))
                batch_items = all_items[start:end]
                batch_seq_ids = seq_ids[start:end]

                async with flight_semaphore:
                    if self.action == "encode":
                        results = await api.encode(
                            items=batch_items, params=self.params
                        )
                    elif self.action == "score":
                        results = await api.score(
                            items=batch_items, params=self.params
                        )
                    else:
                        results = await api.predict(
                            items=batch_items, params=self.params
                        )
                    if not isinstance(results, list):
                        results = [results]

                batch_data = []
                for seq_id, result in zip(batch_seq_ids, results):
                    if (
                        isinstance(result, dict)
                        and "error" in result
                        and "status_code" in result
                    ):
                        err = result["error"]
                        err_str = str(err)
                        seq = seq_id_to_seq.get(seq_id, "?")
                        if (
                            "ACDEFGHIKLMNPQRSTVWY" in err_str
                            or "Residues can only" in err_str
                        ):
                            invalid = sorted(
                                {c for c in str(seq) if c not in "ACDEFGHIKLMNPQRSTVWY"}
                            )
                            print(
                                f"  WARNING: seq_id {seq_id} '{str(seq)[:20]}...' skipped — "
                                f"non-standard residue(s) {invalid} not accepted by {self.model_name}"
                            )
                        else:
                            msg = (
                                "; ".join(
                                    f"{k}: {v[0] if isinstance(v, list) else v}"
                                    for k, v in err.items()
                                )
                                if isinstance(err, dict)
                                else str(err)[:120]
                            )
                            print(f"  WARNING: seq_id {seq_id} skipped — {msg}")
                        continue

                    if self.action in ("predict", "score"):
                        if isinstance(result, dict):
                            for spec in self._resolved:
                                value = self._extract_with_spec(result, spec)
                                batch_data.append(
                                    {
                                        "sequence_id": seq_id,
                                        "prediction_type": spec.column,
                                        "model_name": self.model_name,
                                        "value": value,
                                        "metadata": {"params": self.params},
                                    }
                                )
                    elif self.action == "encode":
                        self._store_embeddings(datastore, seq_id, result)

                if batch_data:
                    datastore.add_predictions_batch(batch_data)

            try:
                await asyncio.gather(
                    *[_dispatch_batch(i) for i in range(n_batches)]
                )
            except Exception as e:
                if self.skip_on_error:
                    print(f"  Error during prediction (skipped): {e}")
                    failed_batch = [
                        {
                            "sequence_id": sid,
                            "prediction_type": self.prediction_type,
                            "model_name": self.model_name,
                            "value": None,
                            "metadata": {
                                "status": "failed",
                                "error": str(e),
                                "params": self.params,
                            },
                        }
                        for sid in df_uncached["sequence_id"].tolist()
                    ]
                    datastore.add_predictions_batch(failed_batch)
                else:
                    print(f"  Error during prediction: {e}")
                    raise

        # --- Vectorized result merge: single JOIN query, not N individual queries ---
        if self.action in ("predict", "score"):
            # Determine which prediction_types to merge
            merge_specs = self._resolved

            # Materialize once — reused across all specs
            all_seq_ids = df["sequence_id"].tolist()

            for spec in merge_specs:
                pred_df = datastore.get_predictions_bulk(
                    all_seq_ids, spec.column, self.model_name
                )
                if not pred_df.empty:
                    # Map sequence_id → value for O(n) assignment instead of merge
                    val_map = dict(zip(pred_df["sequence_id"], pred_df["value"]))
                    df[spec.column] = df["sequence_id"].map(val_map)
                else:
                    df[spec.column] = None

            # Use first extraction's column for the .notna() output filter
            primary_type = merge_specs[0].column
            df_out = df[df[primary_type].notna()].copy()
            filtered_count = len(df) - len(df_out)

        elif self.action == "encode":
            # Find sequences that received embeddings via single DuckDB query
            emb_ids = set(
                datastore.conn.execute(
                    "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ?",
                    [self.model_name],
                )
                .df()["sequence_id"]
                .tolist()
            )
            df_out = df[df["sequence_id"].isin(emb_ids)].copy()
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

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Process sequences using WorkingSet with bounded-concurrency batching.

        **Batching strategy (two levels):**

        This method splits uncached sequences into pipeline-level batches of
        ``batch_size`` items (default 32) and keeps up to ``max_concurrent``
        batches in flight at once.  As each batch completes, its results are
        written to DuckDB immediately and a new batch is dispatched.

        Each pipeline batch is itself an SDK call that may be further split
        by the model's ``maxItems`` schema limit — the SDK handles that
        internally via ``asyncio.gather`` throttled by a semaphore.

        **Why bounded concurrency instead of all-at-once or sequential:**

        - *Sequential* (old behavior) leaves the API idle between batches.
          With 300 sequences and 200ms API latency, that's 10 idle round-trips.
        - *All-at-once* (``process_streaming`` style) has no backpressure —
          10k sequences creates 300+ in-flight tasks with all items and
          response payloads in memory simultaneously.
        - *Bounded concurrency* keeps the API saturated (``max_concurrent``
          batches in flight) while bounding memory to at most
          ``max_concurrent * batch_size`` items plus their results.  DuckDB
          writes happen as each batch lands, providing natural backpressure:
          a slow datastore slows down new batch dispatch.

        Steps:
            1. Cache check via anti-join (DuckDB)
            2. Fetch only uncached (sequence_id, sequence) pairs
            3. Dispatch batches with bounded concurrency
            4. Write results to DuckDB as each batch completes
            5. Return WorkingSet of IDs that have predictions
        """
        input_count = len(ws)
        input_ids = list(ws.sequence_ids)

        # Vectorized cache check
        uncached_ids = datastore.get_uncached_sequence_ids(
            input_ids, self.prediction_type, self.model_name
        )
        cached_count = input_count - len(uncached_ids)

        print(f"  Cached: {cached_count}/{input_count}")
        print(f"  To compute: {len(uncached_ids)}")

        if uncached_ids:
            print(f"  Calling {self.model_name}.{self.action}...")

            # Reuse client
            if self._api_client is None:
                self._api_client = BioLMApiClient(
                    self.model_name, semaphore=self._connection_semaphore, retry_error_batches=True
                )
            api = self._api_client

            # Fetch (id, sequence) pairs — lightweight, no full DataFrame
            id_seq_pairs = datastore.get_sequences_for_ids(uncached_ids)
            seq_id_to_seq = dict(id_seq_pairs)

            all_seq_ids = [p[0] for p in id_seq_pairs]

            # Build items list — either from item_columns or default sequence
            if self.item_columns:
                col_names = list(self.item_columns.values())
                col_map = datastore.get_sequences_for_ids_with_columns(
                    all_seq_ids, col_names
                )
                if not col_map:
                    col_map = datastore.get_sequence_attributes_for_ids(
                        all_seq_ids, col_names
                    )
                all_items = []
                for sid in all_seq_ids:
                    vals = col_map.get(sid, {})
                    item = {}
                    for api_field, col in self.item_columns.items():
                        v = vals.get(col)
                        if v is None or v == "":
                            print(
                                f"  WARNING: seq_id {sid} has "
                                f"NULL/empty value for column "
                                f"'{col}' (API field '{api_field}')"
                            )
                        item[api_field] = v or ""
                    all_items.append(item)
            else:
                all_items = [{"sequence": p[1]} for p in id_seq_pairs]

            # --- Bounded-concurrency dispatch ---
            # Split into pipeline batches, keep max_concurrent in flight.
            batch_size = self.batch_size
            n_batches = (len(all_items) + batch_size - 1) // batch_size
            flight_semaphore = asyncio.Semaphore(self.max_concurrent)

            async def _dispatch_batch(batch_idx):
                """Send one batch to the API and store results in DuckDB."""
                start = batch_idx * batch_size
                end = min(start + batch_size, len(all_items))
                batch_items = all_items[start:end]
                batch_seq_ids = all_seq_ids[start:end]

                async with flight_semaphore:
                    if self.action == "encode":
                        results = await api.encode(
                            items=batch_items, params=self.params
                        )
                    elif self.action == "score":
                        results = await api.score(
                            items=batch_items, params=self.params
                        )
                    else:
                        results = await api.predict(
                            items=batch_items, params=self.params
                        )
                    if not isinstance(results, list):
                        results = [results]

                # Process results and write to DuckDB immediately
                batch_data = []
                for seq_id, result in zip(batch_seq_ids, results):
                    if (
                        isinstance(result, dict)
                        and "error" in result
                        and "status_code" in result
                    ):
                        err = result["error"]
                        err_str = str(err)
                        seq = seq_id_to_seq.get(seq_id, "?")
                        if (
                            "ACDEFGHIKLMNPQRSTVWY" in err_str
                            or "Residues can only" in err_str
                        ):
                            invalid = sorted(
                                {
                                    c
                                    for c in str(seq)
                                    if c not in "ACDEFGHIKLMNPQRSTVWY"
                                }
                            )
                            print(
                                f"  WARNING: seq_id {seq_id} '{str(seq)[:20]}...' "
                                f"skipped — non-standard residue(s) {invalid}"
                            )
                        else:
                            msg = (
                                "; ".join(
                                    f"{k}: {v[0] if isinstance(v, list) else v}"
                                    for k, v in err.items()
                                )
                                if isinstance(err, dict)
                                else str(err)[:120]
                            )
                            print(f"  WARNING: seq_id {seq_id} skipped — {msg}")
                        continue

                    if self.action in ("predict", "score"):
                        if isinstance(result, dict):
                            for spec in self._resolved:
                                value = self._extract_with_spec(result, spec)
                                batch_data.append(
                                    {
                                        "sequence_id": seq_id,
                                        "prediction_type": spec.column,
                                        "model_name": self.model_name,
                                        "value": value,
                                        "metadata": {"params": self.params},
                                    }
                                )
                    elif self.action == "encode":
                        self._store_embeddings(datastore, seq_id, result)

                if batch_data:
                    datastore.add_predictions_batch(batch_data)

            try:
                # Launch all batches — semaphore limits how many run at once.
                # Each batch writes to DuckDB as soon as it completes, so
                # memory holds at most max_concurrent batches of results.
                await asyncio.gather(
                    *[_dispatch_batch(i) for i in range(n_batches)]
                )
                computed = len(uncached_ids)
                print(
                    f"  Completed: {computed} sequences "
                    f"in {n_batches} batches "
                    f"(max {self.max_concurrent} concurrent)"
                )

            except Exception as e:
                if self.skip_on_error:
                    print(f"  Error during prediction (skipped): {e}")
                    failed_batch = [
                        {
                            "sequence_id": sid,
                            "prediction_type": self.prediction_type,
                            "model_name": self.model_name,
                            "value": None,
                            "metadata": {
                                "status": "failed",
                                "error": str(e),
                                "params": self.params,
                            },
                        }
                        for sid in uncached_ids
                    ]
                    datastore.add_predictions_batch(failed_batch)
                else:
                    print(f"  Error during prediction: {e}")
                    raise

        # Return WorkingSet: IDs that now have predictions
        if self.action in ("predict", "score"):
            ids_with_pred = datastore.get_sequence_ids_with_prediction(
                input_ids, self.prediction_type, self.model_name
            )
            ws_out = WorkingSet.from_ids(ids_with_pred)
        elif self.action == "encode":
            emb_ids = set(
                datastore.conn.execute(
                    "SELECT DISTINCT sequence_id FROM embeddings WHERE model_name = ?",
                    [self.model_name],
                )
                .df()["sequence_id"]
                .tolist()
            )
            ws_out = WorkingSet(
                frozenset(sid for sid in ws.sequence_ids if sid in emb_ids)
            )
        else:
            ws_out = ws

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            cached_count=cached_count,
            computed_count=len(uncached_ids),
            filtered_count=input_count - len(ws_out),
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
            filter_func,
            "requires_complete_data",
            True,  # Default: safe (batch)
        )

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Apply filter to DataFrame and return the filtered result."""
        start_count = len(df)

        print(f"  Applying filter: {self.filter_func}")
        if df.empty:
            df_filtered = df
        else:
            df_filtered = self.filter_func(
                df
            ).copy()  # .copy() prevents in-place mutation of input df

        filtered_count = start_count - len(df_filtered)

        print(f"  Filtered out: {filtered_count}/{start_count}")
        print(f"  Remaining: {len(df_filtered)}")

        # Persist filter results for resume support
        run_id = kwargs.get("run_id")
        if run_id and "sequence_id" in df_filtered.columns:
            datastore.save_filter_results(
                run_id, self.name, df_filtered["sequence_id"].tolist()
            )

        return df_filtered, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df_filtered),
            filtered_count=filtered_count,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Apply filter using WorkingSet.

        Two execution paths — chosen at call time, not as a "fallback":

        1. **SQL-native** (zero materialization): filters that implement
           ``to_sql()`` return a complete SELECT scoped to the working set.
           DuckDB executes it directly; no DataFrame is ever created.

        2. **DataFrame-based**: filters that *cannot* be expressed in SQL
           (e.g. HammingDistanceFilter, CustomFilter) must materialize a
           DataFrame.  This is the correct path for those filters, not a
           fallback.
        """
        input_count = len(ws)

        print(f"  Applying filter: {self.filter_func}")

        if not ws:
            return ws, StageResult(
                stage_name=self.name,
                input_count=0,
                output_count=0,
                filtered_count=0,
            )

        # Determine execution path: SQL-native or DataFrame-based
        sql_query = None
        if isinstance(self.filter_func, BaseFilter):
            sql_query = self.filter_func.to_sql()

        if sql_query is not None:
            # SQL-native path — zero materialization
            surviving_ids = datastore.execute_filter_sql(
                list(ws.sequence_ids), sql_query
            )
            ws_out = WorkingSet.from_ids(surviving_ids)
        else:
            # DataFrame-based path — required for non-SQL filters
            df = datastore.materialize_working_set(ws)
            df_filtered = self.filter_func(df).copy()
            if "sequence_id" in df_filtered.columns:
                ws_out = WorkingSet.from_ids(df_filtered["sequence_id"].tolist())
            else:
                ws_out = ws

        filtered_count = input_count - len(ws_out)

        print(f"  Filtered out: {filtered_count}/{input_count}")
        print(f"  Remaining: {len(ws_out)}")

        # Persist filter results for resume support
        run_id = kwargs.get("run_id")
        if run_id:
            datastore.save_filter_results(
                run_id, self.name, list(ws_out.sequence_ids)
            )

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            filtered_count=filtered_count,
        )


class CofoldingPredictionStage(Stage):
    """
    Prediction stage for co-folding models (Boltz2, Chai-1).

    Each sequence in the pipeline DataFrame is used as the **primary** entity
    (chain) in a multi-molecule folding request.  Additional static entities —
    ligands, cofactors, DNA/RNA, or other protein chains — are injected via
    ``static_entities`` and held constant for every sequence.

    The caller is responsible for providing the correct molecule field names and
    params for the target model.  Boltz2 and Chai-1 both expect an item shaped
    like ``{'molecules': [{'id': ..., 'type': ..., 'sequence': ...}, ...]}``.

    Args:
        name: Stage name.
        model_name: BioLM model slug (``'boltz2'``, ``'chai1'``).
        action: API action (default ``'predict'``).
        prediction_type: Column name written for the confidence score
            (default ``'structure'``).
        sequence_chain_id: Chain ID / name assigned to the pipeline's primary
            sequence in the molecules list.
        sequence_entity_type: Molecule type for the primary sequence
            (``'protein'``, ``'dna'``, ``'rna'``).
        static_entities: List of :class:`~biolmai.pipeline.generative.FoldingEntity`
            objects appended to every molecules list after the primary chain.
        params: Model-specific params dict passed directly to the API
            (e.g. ``{'recycling_steps': 3, 'sampling_steps': 20}`` for Boltz).
        batch_size: Sequences per API call (default 1; co-folding models are
            typically limited to batch size 1).

    Example::

        from biolmai.pipeline import FoldingEntity

        pipeline.add_cofolding_prediction(
            model_name='boltz2',
            static_entities=[
                FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
            ],
            params={'recycling_steps': 3, 'sampling_steps': 20},
            depends_on=['filter_top50'],
        )
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        action: str = "predict",
        prediction_type: str = "structure",
        sequence_chain_id: str = "A",
        sequence_entity_type: str = "protein",
        static_entities=None,  # List[FoldingEntity] — typed at call site
        params: Optional[dict] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(name=name, model_name=model_name, **kwargs)
        self.action = action
        self.prediction_type = prediction_type
        self.sequence_chain_id = sequence_chain_id
        self.sequence_entity_type = sequence_entity_type
        self.static_entities = static_entities or []
        self.params = params or {}
        self.batch_size = batch_size
        self._api_client = None

    def _build_item(self, sequence: str) -> dict:
        """Build the multi-molecule item dict for a single pipeline sequence."""
        molecules = [
            {
                "id": self.sequence_chain_id,
                "type": self.sequence_entity_type,
                "sequence": sequence,
            }
        ]
        for entity in self.static_entities:
            mol: dict[str, Any] = {"id": entity.id, "type": entity.entity_type}
            if entity.sequence is not None:
                mol["sequence"] = entity.sequence
            if entity.smiles is not None:
                mol["smiles"] = entity.smiles
            if entity.ccd is not None:
                mol["ccd"] = entity.ccd
            molecules.append(mol)
        return {"molecules": molecules}

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Run co-folding prediction for each sequence in *df*."""
        start_count = len(df)
        computed = 0
        df = df.copy()

        if self._api_client is None:
            self._api_client = BioLMApiClient(self.model_name)
        api = self._api_client

        for i in range(0, len(df), self.batch_size):
            batch = df.iloc[i : i + self.batch_size]
            items = [
                self._build_item(row["sequence"]) for _, row in batch.iterrows()
            ]

            results = await getattr(api, self.action)(
                items=items, params=self.params
            )

            for _j, (result, (idx, row)) in enumerate(
                zip(results, batch.iterrows())
            ):
                if not isinstance(result, dict):
                    continue
                cif = result.get("cif", "")
                conf_data = result.get("confidence", {})
                confidence = (
                    conf_data.get("confidence_score")
                    if isinstance(conf_data, dict)
                    else None
                )
                seq_id = row.get("sequence_id")
                if seq_id is not None and cif:
                    datastore.add_structure(
                        seq_id,
                        model_name=self.model_name,
                        structure_str=cif,
                        format="cif",
                    )
                if seq_id is not None and confidence is not None:
                    datastore.add_prediction(
                        seq_id,
                        self.prediction_type,
                        self.model_name,
                        float(confidence),
                        metadata={"cif_stored": bool(cif)},
                    )
                df.at[idx, self.prediction_type] = confidence
                if cif:
                    df.at[idx, "cif"] = cif
                computed += 1

        return df, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=len(df),
            computed_count=computed,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Run co-folding prediction using WorkingSet — no DataFrame transport."""
        input_count = len(ws)
        computed = 0

        id_seq_pairs = datastore.get_sequences_for_ids(list(ws.sequence_ids))

        if self._api_client is None:
            self._api_client = BioLMApiClient(self.model_name)
        api = self._api_client

        for i in range(0, len(id_seq_pairs), self.batch_size):
            chunk = id_seq_pairs[i : i + self.batch_size]
            items = [self._build_item(seq) for _sid, seq in chunk]

            results = await getattr(api, self.action)(
                items=items, params=self.params
            )

            for (seq_id, _seq), result in zip(chunk, results):
                if not isinstance(result, dict):
                    continue
                cif = result.get("cif", "")
                conf_data = result.get("confidence", {})
                confidence = (
                    conf_data.get("confidence_score")
                    if isinstance(conf_data, dict)
                    else None
                )
                if cif:
                    datastore.add_structure(
                        seq_id,
                        model_name=self.model_name,
                        structure_str=cif,
                        format="cif",
                    )
                if confidence is not None:
                    datastore.add_prediction(
                        seq_id,
                        self.prediction_type,
                        self.model_name,
                        float(confidence),
                        metadata={"cif_stored": bool(cif)},
                    )
                computed += 1

        # Return IDs that got predictions
        ids_with_pred = datastore.get_sequence_ids_with_prediction(
            list(ws.sequence_ids), self.prediction_type, self.model_name
        )
        ws_out = WorkingSet.from_ids(ids_with_pred) if ids_with_pred else ws

        return ws_out, StageResult(
            stage_name=self.name,
            input_count=input_count,
            output_count=len(ws_out),
            computed_count=computed,
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
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_metric: str = "hamming",
        embedding_model: Optional[str] = None,
        max_sample: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.method = method
        self.n_clusters = n_clusters
        self.similarity_metric = similarity_metric
        self.embedding_model = embedding_model
        self.max_sample = max_sample
        self.cluster_kwargs = kwargs

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Cluster sequences, add cluster columns, and return the enriched DataFrame."""
        import warnings

        from biolmai.pipeline.clustering import (
            LARGE_DATASET_THRESHOLD,
            VERY_LARGE_DATASET_THRESHOLD,
            SequenceClusterer,
        )

        start_count = len(df)
        df = df.copy()

        n = start_count
        if self.similarity_metric == "hamming":
            if n > VERY_LARGE_DATASET_THRESHOLD:
                raise ValueError(
                    f"ClusteringStage: {n} sequences with hamming distance requires "
                    f"~{n**2 // 1_000_000}M comparisons. Use similarity_metric='embedding' "
                    f"or max_sample<={VERY_LARGE_DATASET_THRESHOLD}."
                )
            elif n > LARGE_DATASET_THRESHOLD:
                warnings.warn(
                    f"ClusteringStage: {n} sequences with hamming distance is O(n²) expensive. "
                    f"Consider similarity_metric='embedding' or max_sample={LARGE_DATASET_THRESHOLD}.",
                    ResourceWarning,
                    stacklevel=2,
                )

        print(f"  Clustering {start_count} sequences using {self.method}...")

        sequences = df["sequence"].tolist()

        # Load embeddings if needed — bulk fetch, not per-sequence
        embeddings = None
        if self.similarity_metric == "embedding":
            if self.embedding_model is None:
                raise ValueError(
                    "embedding_model required when similarity_metric='embedding'"
                )

            print(f"  Loading embeddings from {self.embedding_model}...")
            # Bug #6 fix: bulk fetch in a single JOIN instead of N per-sequence queries
            seq_ids = df["sequence_id"].tolist()
            emb_map = datastore.get_embeddings_bulk(
                seq_ids, model_name=self.embedding_model
            )
            embeddings_list = []
            for seq_id, seq in zip(seq_ids, sequences):
                emb_array = emb_map.get(int(seq_id))
                if emb_array is None:
                    raise ValueError(
                        f"No embedding found for sequence_id {seq_id} ({seq[:20]}...)"
                    )
                embeddings_list.append(emb_array)

            embeddings = np.stack(embeddings_list)

        # Perform clustering
        clusterer_kwargs = dict(self.cluster_kwargs.items())
        if self.max_sample is not None:
            clusterer_kwargs["max_sample"] = self.max_sample
        clusterer = SequenceClusterer(
            method=self.method,
            n_clusters=self.n_clusters,
            similarity_metric=self.similarity_metric,
            **clusterer_kwargs,
        )

        result = clusterer.cluster(sequences, embeddings)

        # Add cluster assignments to output DataFrame
        df["cluster_id"] = result.cluster_ids
        df["is_centroid"] = False
        df.loc[result.centroid_indices, "is_centroid"] = True

        print(f"  Found {result.n_clusters} clusters")
        if result.silhouette_score is not None:
            print(f"  Silhouette score: {result.silhouette_score:.3f}")
        if result.davies_bouldin_score is not None:
            print(f"  Davies-Bouldin score: {result.davies_bouldin_score:.3f}")

        # Store summary metadata
        meta = {
            "method": self.method,
            "n_clusters": result.n_clusters,
            "silhouette_score": result.silhouette_score,
            "davies_bouldin_score": result.davies_bouldin_score,
            "cluster_sizes": result.cluster_sizes,
        }
        datastore.set_pipeline_metadata(f"clustering_{self.name}", meta)

        # Store per-sequence assignments for resume support
        assignments = {
            seq: int(cid) for seq, cid in zip(sequences, result.cluster_ids.tolist())
        }
        datastore.set_pipeline_metadata(
            f"clustering_{self.name}_assignments", assignments
        )

        return df, StageResult(
            stage_name=self.name,
            input_count=start_count,
            output_count=start_count,
            computed_count=start_count,
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Cluster sequences — materializes internally for scikit-learn.

        Clustering doesn't filter rows (all sequences survive), so the
        returned WorkingSet is the same as the input.  Cluster assignments
        are stored as predictions (prediction_type ``'cluster_id'``) so that
        ``materialize_working_set()`` includes them in ``get_final_data()``.
        """
        df = datastore.materialize_working_set(ws, include_predictions=False)
        df_out, result = await self.process(df, datastore, **kwargs)

        # Store cluster_id as predictions so get_final_data() picks them up
        if "cluster_id" in df_out.columns and "sequence_id" in df_out.columns:
            batch = [
                {
                    "sequence_id": int(row["sequence_id"]),
                    "prediction_type": "cluster_id",
                    "model_name": self.name,
                    "value": float(row["cluster_id"]),
                }
                for _, row in df_out.iterrows()
                if row["cluster_id"] is not None
            ]
            if batch:
                datastore.add_predictions_batch(batch)

        return ws, result


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
        sequences: Union[list[str], pd.DataFrame, str, Path] = None,
        diff_mode: bool = False,
        input_columns: Optional[list[str]] = None,
        **kwargs,
    ):
        # Build InputSchema if input_columns specified
        if input_columns is not None:
            kwargs["input_schema"] = InputSchema(columns=input_columns)
        super().__init__(**kwargs)
        self.input_sequences = sequences
        self.diff_mode = diff_mode
        self.input_columns = input_columns

    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """Load sequences into DataFrame."""

        if self.input_sequences is None:
            raise ValueError("No sequences provided. Set 'sequences' parameter.")

        # Convert to DataFrame
        if isinstance(self.input_sequences, list):
            df = pd.DataFrame({"sequence": self.input_sequences})

        elif isinstance(self.input_sequences, pd.DataFrame):
            df = self.input_sequences.copy()

        elif isinstance(self.input_sequences, (str, Path)):
            path = Path(self.input_sequences)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix == ".csv":
                df = pd.read_csv(path)

            elif path.suffix in [".fasta", ".fa", ".faa"]:
                # Simple FASTA parser
                sequences = []
                current_seq = []

                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(">"):
                            if current_seq:
                                sequences.append("".join(current_seq))
                                current_seq = []
                        else:
                            current_seq.append(line)

                    if current_seq:
                        sequences.append("".join(current_seq))

                if not sequences:
                    raise ValueError(f"No sequences found in FASTA file: {path}")
                df = pd.DataFrame({"sequence": sequences})

            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        else:
            raise TypeError(
                f"Unsupported type for sequences: {type(self.input_sequences)}"
            )

        # ----- Schema compatibility validation -----------------------------------
        ds_has_data = self.datastore.conn.execute(
            "SELECT COUNT(*) FROM sequences"
        ).fetchone()[0] > 0

        if ds_has_data:
            ds_extra = self.datastore._extra_columns
            if self.input_columns is not None and not ds_extra:
                raise ValueError(
                    "Cannot use multi-column input (input_columns="
                    f"{self.input_columns}) on a datastore that already "
                    "contains single-column sequences. Use a fresh datastore "
                    "or match the existing schema."
                )
            if self.input_columns is None and ds_extra:
                raise ValueError(
                    "Cannot use single-column input on a datastore that "
                    f"already contains multi-column data (columns: "
                    f"{sorted(ds_extra)}). Use a fresh datastore or provide "
                    "input_columns matching the existing schema."
                )

        # ----- Multi-column input path ------------------------------------------
        if self.input_columns is not None:
            # Validate that all input columns are present
            missing = [c for c in self.input_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"input_columns {missing} not found in DataFrame. "
                    f"Available: {list(df.columns)}"
                )

            # Ensure the sequences table has the extra columns
            self.datastore.ensure_input_columns(self.input_columns)

            # Deduplicate across all input columns
            initial_count = len(df)
            df = df.drop_duplicates(subset=self.input_columns).reset_index(drop=True)
            deduplicated_count = initial_count - len(df)
            if deduplicated_count > 0 and self.verbose:
                print(f"Deduplicated {deduplicated_count} rows ({len(df)} unique)")

            # Use the multi-column add_sequences_batch path
            df["sequence_id"] = self.datastore.add_sequences_batch(
                input_df=df, input_columns=self.input_columns
            )

            # Also persist non-input extra columns as sequence attributes
            extra_cols = [
                c
                for c in df.columns
                if c not in self.input_columns
                and c not in ("sequence", "sequence_id", "length", "hash")
            ]
            if extra_cols:
                seq_ids = df["sequence_id"].tolist()
                for col in extra_cols:
                    self.datastore.store_sequence_attributes(
                        seq_ids, col, df[col].tolist()
                    )

            return df

        # ----- Legacy single-column path ----------------------------------------
        if "sequence" not in df.columns:
            raise ValueError("DataFrame must have 'sequence' column")

        # Deduplicate before inserting into datastore
        initial_count = len(df)
        df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        deduplicated_count = initial_count - len(df)
        if deduplicated_count > 0 and self.verbose:
            print(f"Deduplicated {deduplicated_count} sequences ({len(df)} unique)")

        # Batch-add all sequences in one vectorized call instead of N individual calls
        df["sequence_id"] = self.datastore.add_sequences_batch(df["sequence"].tolist())

        # Persist extra columns (e.g. heavy_chain, light_chain) as sequence attributes
        # so they're available to stages that need them (item_columns in PredictionStage)
        extra_cols = [
            c for c in df.columns if c not in ("sequence", "sequence_id", "length", "hash")
        ]
        if extra_cols:
            seq_ids = df["sequence_id"].tolist()
            for col in extra_cols:
                self.datastore.store_sequence_attributes(
                    seq_ids, col, df[col].tolist()
                )

        # In diff mode, report how many sequences are new vs. already cached
        if self.diff_mode:
            existing_count = self.datastore.count_matching_sequences(
                df["sequence"].tolist()
            )
            new_count = len(df) - existing_count
            if self.verbose:
                print(f"Diff mode: {existing_count} sequences already in datastore")
                print(f"Diff mode: {new_count} new sequences to process")

        return df

    def get_merged_results(
        self,
        prediction_types: Optional[list[str]] = None,
        sequence_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get results merged with existing cached data (for diff mode).

        This method is SQL-based and efficient - it doesn't load millions of rows.
        Instead, it queries only the data you need using DuckDB's columnar engine.

        NOTE (Bug #5): ``sequence_filter`` and ``sql_where`` in query_results() are
        interpolated directly into SQL.  These parameters are intended for internal/
        trusted caller use only — never pass untrusted user input to them.

        Args:
            prediction_types: List of prediction types to include (None = all)
            sequence_filter: SQL WHERE clause to filter sequences (e.g., "length > 50").
                             TRUSTED CALLERS ONLY — not safe for untrusted user input.

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
            seq_ids_df = pd.DataFrame({"sequence_id": df["sequence_id"].tolist()})
            self.datastore.conn.register("_merged_seq_ids", seq_ids_df)
            try:
                pred_types_df = self.datastore.conn.execute(
                    """
                    SELECT DISTINCT p.prediction_type
                    FROM predictions p
                    INNER JOIN _merged_seq_ids m ON p.sequence_id = m.sequence_id
                """
                ).df()
            finally:
                self.datastore.conn.unregister("_merged_seq_ids")

            pred_types = pred_types_df["prediction_type"].tolist()
            for pred_type in pred_types:
                self._add_predictions_to_df(df, pred_type)

        if self.verbose:
            print(f"\nDiff mode: Loaded {len(df)} sequences with predictions")

        return df

    def _add_predictions_to_df(self, df: pd.DataFrame, prediction_type: str):
        """
        Efficiently add a prediction column to DataFrame using DuckDB SQL.
        Modifies df in place by leveraging DuckDB's join capabilities.
        """
        if len(df) == 0:
            return

        # Use DuckDB to efficiently join predictions — register df explicitly first
        self.datastore.conn.register("_pred_df", df)
        try:
            result = self.datastore.conn.execute(
                """
                SELECT
                    _pred_df.sequence_id,
                    p.value
                FROM _pred_df
                LEFT JOIN predictions p
                    ON _pred_df.sequence_id = p.sequence_id
                    AND p.prediction_type = ?
            """,
                [prediction_type],
            ).df()
        finally:
            self.datastore.conn.unregister("_pred_df")

        # Create dict for fast lookup
        pred_dict = dict(zip(result["sequence_id"], result["value"]))

        # Add column to DataFrame
        df[prediction_type] = df["sequence_id"].map(pred_dict)

    def query_results(
        self, sql_where: str, columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Query results using SQL WHERE clause (for diff mode with large datasets).

        This leverages DuckDB's vectorized engine for maximum performance.

        NOTE (Bug #5): ``sql_where`` is interpolated directly into SQL.
        TRUSTED CALLERS ONLY — never pass untrusted user input to this parameter.

        Args:
            sql_where: SQL WHERE clause using table aliases:
                      - s.* for sequences table (e.g., "s.length > 100")
                      - Column names directly (no p. prefix needed)
                      TRUSTED CALLERS ONLY — not safe for untrusted user input.
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

    # ------------------------------------------------------------------
    # Data exploration helpers
    # ------------------------------------------------------------------

    def explore(self) -> dict[str, Any]:
        """Return summary stats for the pipeline's datastore (all via SQL).

        Returns:
            Dict with keys: sequences, embeddings, generated, completed_stages,
            predictions (dict of prediction_type → count).
        """
        row = self.datastore.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM sequences)           AS seq_n,
                (SELECT COUNT(*) FROM embeddings)          AS emb_n,
                (SELECT COUNT(*) FROM generation_metadata) AS gen_n,
                (SELECT COUNT(*) FROM stage_completions
                 WHERE status = 'completed')               AS done_stages
        """
        ).fetchone()

        pred_rows = self.datastore.conn.execute(
            """
            SELECT prediction_type, COUNT(*) AS n
            FROM predictions
            GROUP BY prediction_type
        """
        ).fetchall()

        return {
            "sequences": int(row[0]),
            "embeddings": int(row[1]),
            "generated": int(row[2]),
            "completed_stages": int(row[3]),
            "predictions": {r[0]: int(r[1]) for r in pred_rows},
        }

    def stats(self, stage_name: Optional[str] = None) -> pd.DataFrame:
        """Return per-stage counts from stage_completions.

        Args:
            stage_name: If provided, filter to that stage only.

        Returns:
            DataFrame with columns: stage_name, status, input_count,
            output_count, completed_at.
        """
        if stage_name:
            return self.datastore.query(
                """
                SELECT stage_name, status, input_count, output_count, completed_at
                FROM stage_completions
                WHERE stage_name = ?
                ORDER BY completed_at
            """,
                [stage_name],
            )
        return self.datastore.query(
            """
            SELECT stage_name, status, input_count, output_count, completed_at
            FROM stage_completions
            ORDER BY completed_at
        """
        )

    def query(self, sql: str, params=None) -> pd.DataFrame:
        """Execute arbitrary SQL against the pipeline's DuckDB datastore.

        Args:
            sql: DuckDB SQL query string.
            params: Optional list of query parameters.

        Returns:
            DataFrame with results.

        Example:
            >>> pipeline.query("SELECT * FROM sequences WHERE length > 100")
        """
        return self.datastore.query(sql, params)

    def plot(self, kind: str = "funnel", **kwargs):
        """Convenience wrapper around PipelinePlotter.

        Args:
            kind: One of 'funnel', 'predictions', 'distributions'.
            **kwargs: Forwarded to the underlying plotter method.
        """
        from biolmai.pipeline.visualization import PipelinePlotter

        plotter = PipelinePlotter(self)
        if kind == "funnel":
            return plotter.plot_funnel(self.stage_results, **kwargs)
        elif kind == "predictions":
            return plotter.plot_predictions(**kwargs)
        elif kind == "distributions":
            return plotter.plot_distributions(**kwargs)
        else:
            raise ValueError(
                f"Unknown plot kind '{kind}'. Choose: 'funnel', 'predictions', 'distributions'"
            )

    def add_prediction(
        self,
        model_name: str,
        action: str = "predict",
        extractions: Optional[Union[str, list[Union[str, ExtractionSpec]]]] = None,
        columns: Optional[Union[str, dict[str, str]]] = None,
        params: Optional[dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add a prediction stage.

        Args:
            model_name: BioLM model name
            action: API action ('predict', 'encode', 'score')
            extractions: API response key(s) to extract. Required for
                predict/score actions. Can be a string for a single key or a
                list of strings / ExtractionSpec objects.
            columns: Output column name(s). A string renames a single
                extraction; a dict maps response keys to column names
                (unmapped keys keep their name).
            params: Optional API parameters
            stage_name: Custom stage name (defaults to ``predict_{first_column}``)
            depends_on: List of stage names this depends on

        Example::

            pipeline.add_prediction(
                "temberture-regression",
                extractions="prediction",
                columns="tm",
            )
        """
        # Auto-derive stage name from first column
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
            >>> pipeline.add_predictions([
            ...     {'model_name': 'temberture-regression', 'extractions': 'prediction', 'columns': 'tm'},
            ...     {'model_name': 'soluprot', 'extractions': 'soluble', 'columns': 'solubility'},
            ... ])
        """
        for model in models:
            if isinstance(model, str):
                self.add_prediction(
                    model_name=model, action=action, depends_on=depends_on, **kwargs
                )
            elif isinstance(model, dict):
                model_config = {**kwargs, **model}
                model_config["depends_on"] = model_config.get("depends_on", depends_on)
                model_name = model_config.pop("model_name")
                self.add_prediction(model_name=model_name, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")

        return self

    def add_filter(
        self,
        filter_func: Union[BaseFilter, callable],
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
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
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_clustering(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        similarity_metric: str = "hamming",
        embedding_model: Optional[str] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
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
        static_entities=None,  # List[FoldingEntity]
        depends_on: Optional[list[str]] = None,
        params: Optional[dict] = None,
        batch_size: int = 1,
    ):
        """
        Add a co-folding prediction stage (Boltz2, Chai-1).

        Each pipeline sequence becomes the primary entity in a multi-molecule
        folding request.  ``static_entities`` injects ligands, cofactors,
        DNA/RNA strands, or additional protein chains that are constant across
        all designs.

        The caller is responsible for providing the correct molecule field names
        via ``static_entities`` and the right ``params`` for the model.

        Args:
            model_name: BioLM model slug (``'boltz2'``, ``'chai1'``).
            action: API action (default ``'predict'``).
            stage_name: Optional stage name (defaults to ``model_name``).
            prediction_type: Column name for the confidence score.
            sequence_chain_id: Chain ID assigned to the primary sequence
                (e.g. ``'A'`` for Boltz, molecule ``name`` for Chai-1).
            sequence_entity_type: Entity type for the primary sequence
                (``'protein'``, ``'dna'``, ``'rna'``).
            static_entities: List of :class:`FoldingEntity` objects to include
                in every request alongside the primary sequence.
            depends_on: Upstream stage names.
            params: Model-specific params (e.g. ``{'recycling_steps': 3}``).
            batch_size: Sequences per API call (default 1).

        Example::

            from biolmai.pipeline import FoldingEntity

            pipeline.add_cofolding_prediction(
                model_name='boltz2',
                static_entities=[
                    FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
                ],
                params={'recycling_steps': 3, 'sampling_steps': 20},
                depends_on=['filter_top50'],
            )
        """
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
        action: str = "predict",
        sequences: Union[list[str], pd.DataFrame, str, Path] = None,
        params: Optional[dict] = None,
        extractions=None,
        columns=None,
        embedding_extractor=None,
        **kwargs,
    ):
        super().__init__(sequences=sequences, **kwargs)

        # Automatically add single prediction stage
        self.add_prediction(
            model_name=model_name,
            action=action,
            params=params,
            extractions=extractions,
            columns=columns,
            embedding_extractor=embedding_extractor,
        )


# Convenience aliases
def Predict(
    model_name: str,
    sequences: Union[list[str], pd.DataFrame, str, Path],
    params: Optional[dict] = None,
    **kwargs,
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
        action="predict",
        sequences=sequences,
        params=params,
        **kwargs,
    )
    pipeline.run()
    return pipeline.get_final_data()


def Embed(
    model_name: str,
    sequences: Union[list[str], pd.DataFrame, str, Path],
    layer: Optional[int] = None,
    **kwargs,
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
    params = {"layer": layer} if layer is not None else None

    pipeline = SingleStepPipeline(
        model_name=model_name,
        action="encode",
        sequences=sequences,
        params=params,
        embedding_extractor=EmbeddingSpec(key="embedding"),
        **kwargs,
    )
    pipeline.run()
    df = pipeline.get_final_data()

    # Load embeddings via bulk fetch (single DuckDB JOIN + batch Parquet reads)
    # instead of N per-sequence queries
    seq_ids = df["sequence_id"].tolist()
    emb_map = pipeline.datastore.get_embeddings_bulk(seq_ids, model_name=model_name)
    df["embedding"] = [emb_map.get(int(sid)) for sid in seq_ids]

    return df
