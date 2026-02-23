"""
Base Pipeline classes for stage management and execution.
"""

import asyncio
import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union, Set, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from biolmai.pipeline.datastore_duckdb import DuckDBDataStore as DataStore


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        cache_key: Key for caching results (prediction_type for predictions)
        depends_on: List of stage names this stage depends on
        model_name: Model name for predictions/structures
        max_concurrent: Maximum concurrent API calls (for rate limiting)
    """
    
    def __init__(
        self,
        name: str,
        cache_key: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        max_concurrent: int = 10
    ):
        self.name = name
        self.cache_key = cache_key or name
        self.depends_on = depends_on or []
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    @abstractmethod
    async def process(
        self,
        df: pd.DataFrame,
        datastore: DataStore,
        **kwargs
    ) -> StageResult:
        """
        Process data through this stage.
        
        Args:
            df: Input DataFrame (must have 'sequence' and 'sequence_id' columns)
            datastore: DataStore for caching
            **kwargs: Additional arguments
        
        Returns:
            StageResult with statistics
        """
        pass
    
    def __repr__(self):
        deps = f", depends_on={self.depends_on}" if self.depends_on else ""
        return f"{self.__class__.__name__}('{self.name}'{deps})"


class BasePipeline(ABC):
    """
    Base class for all pipeline types.
    
    Provides:
    - Stage management and dependency resolution
    - Async execution with progress tracking
    - Caching and resumability
    - Export and visualization
    
    Args:
        datastore: DataStore instance for caching
        run_id: Unique run identifier (auto-generated if not provided)
        output_dir: Directory for outputs
        resume: Whether to resume from previous run
        verbose: Enable verbose output
    """
    
    def __init__(
        self,
        datastore: Optional[Union[DataStore, str, Path]] = None,
        run_id: Optional[str] = None,
        output_dir: Union[str, Path] = 'pipeline_outputs',
        resume: bool = False,
        verbose: bool = True
    ):
        # Setup datastore
        if isinstance(datastore, DataStore):
            self.datastore = datastore
        elif isinstance(datastore, (str, Path)):
            self.datastore = DataStore(datastore)
        else:
            # Auto-create datastore in output_dir
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            db_path = output_path / 'pipeline.db'
            data_dir = output_path / 'pipeline_data'
            self.datastore = DataStore(db_path, data_dir)
        
        self.run_id = run_id or self._generate_run_id()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.verbose = verbose
        
        # Stage management
        self.stages: List[Stage] = []
        self.stage_results: Dict[str, StageResult] = {}
        self._stage_data: Dict[str, pd.DataFrame] = {}  # Cache for stage outputs
        
        # Pipeline state
        self.pipeline_type = self.__class__.__name__
        self.status = 'initialized'
        self.start_time = None
        self.end_time = None
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        short_uuid = str(uuid.uuid4())[:8]
        return f'{timestamp}_{short_uuid}'
    
    def add_stage(self, stage: Stage):
        """Add a stage to the pipeline."""
        # Validate dependencies
        for dep in stage.depends_on:
            if dep not in [s.name for s in self.stages]:
                raise ValueError(
                    f"Stage '{stage.name}' depends on '{dep}' which hasn't been added yet"
                )
        
        self.stages.append(stage)
        if self.verbose:
            print(f"Added stage: {stage}")
    
    def _resolve_dependencies(self) -> List[List[Stage]]:
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
            current_level = [stage_map[name] for name, degree in in_degree.items() if degree == 0]
            
            if not current_level:
                remaining = list(in_degree.keys())
                raise ValueError(f"Circular dependency detected among stages: {remaining}")
            
            levels.append(current_level)
            
            # Remove current level from graph
            for stage in current_level:
                del in_degree[stage.name]
            
            # Decrease in-degree for dependent stages
            for stage in current_level:
                for other_stage in self.stages:
                    if stage.name in other_stage.depends_on and other_stage.name in in_degree:
                        in_degree[other_stage.name] -= 1
        
        return levels
    
    async def _execute_stage(
        self,
        stage: Stage,
        df_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, StageResult]:
        """Execute a single stage."""
        start_time = time.time()
        
        # Check if stage is already complete (resumability)
        stage_id = f"{self.run_id}_{stage.name}"
        if self.resume and self.datastore.is_stage_complete(stage_id):
            if self.verbose:
                print(f"\n✓ Stage '{stage.name}' already complete (resuming)")
            
            # Load cached result
            # This is a simplified version - in practice, we'd need to retrieve the actual data
            result = StageResult(
                stage_name=stage.name,
                input_count=len(df_input),
                output_count=len(df_input),  # Placeholder
                elapsed_time=0
            )
            return df_input, result
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Stage: {stage.name}")
            print(f"Input: {len(df_input):,} sequences")
            if stage.depends_on:
                print(f"Depends on: {', '.join(stage.depends_on)}")
        
        # Execute stage
        result = await stage.process(df_input, self.datastore)
        result.elapsed_time = time.time() - start_time
        
        # Mark stage complete
        self.datastore.mark_stage_complete(
            stage_id=stage_id,
            run_id=self.run_id,
            stage_name=stage.name,
            input_count=result.input_count,
            output_count=result.output_count,
            status='completed'
        )
        
        if self.verbose:
            print(f"\n{result}")
            print(f"{'='*60}")
        
        return df_input, result
    
    async def run_async(self, enable_streaming: bool = False, **kwargs) -> Dict[str, StageResult]:
        """
        Run the pipeline asynchronously.
        
        Args:
            enable_streaming: If True, stream results through per-sequence filters
                            for better parallelism and lower latency.
        
        Returns:
            Dict mapping stage names to StageResults
        """
        self.start_time = time.time()
        self.status = 'running'
        
        # Create pipeline run record
        config = self._get_config()
        self.datastore.create_pipeline_run(
            run_id=self.run_id,
            pipeline_type=self.pipeline_type,
            config=config,
            status='running'
        )
        
        try:
            # Get initial data
            df_current = await self._get_initial_data(**kwargs)
            
            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"# Pipeline: {self.pipeline_type}")
                print(f"# Run ID: {self.run_id}")
                print(f"# Initial sequences: {len(df_current):,}")
                if enable_streaming:
                    print(f"# Streaming: ENABLED")
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
                        enable_streaming and
                        next_stage is not None and
                        next_stage.name not in processed_stages and
                        hasattr(stage, 'process_streaming') and
                        self._can_stream_to_next(stage, next_stage)
                    )
                    
                    if can_stream:
                        # STREAMING: Process and pass results incrementally
                        df_out = await self._execute_stage_streaming(stage, next_stage, df_current)
                        self._stage_data[stage.name] = df_out
                        self._stage_data[next_stage.name] = df_out
                        df_current = df_out
                        # Mark both stages as processed
                        processed_stages.add(stage.name)
                        processed_stages.add(next_stage.name)
                    else:
                        # BATCHING: Wait for complete results
                        df_out, result = await self._execute_stage(stage, df_current)
                        self.stage_results[stage.name] = result
                        self._stage_data[stage.name] = df_out
                        df_current = df_out
                        processed_stages.add(stage.name)
                else:
                    # Multiple stages in level - execute in parallel
                    if self.verbose:
                        print(f"\nExecuting {len(level_stages)} stages in parallel...")
                    
                    tasks = [
                        self._execute_stage(stage, df_current)
                        for stage in level_stages
                    ]
                    results = await asyncio.gather(*tasks)
                    
                    for stage, (df_out, result) in zip(level_stages, results):
                        self.stage_results[stage.name] = result
                        self._stage_data[stage.name] = df_out
                    
                    # For parallel stages, use the output from the last stage
                    # (In practice, you might want more sophisticated merging)
                    df_current = results[-1][0]
            
            self.status = 'completed'
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, 'completed')
            
            if self.verbose:
                total_time = self.end_time - self.start_time
                print(f"\n{'#'*60}")
                print(f"# Pipeline completed in {total_time:.1f}s")
                print(f"# Final sequences: {len(df_current):,}")
                print(f"{'#'*60}\n")
            
            return self.stage_results
        
        except Exception as e:
            self.status = 'failed'
            self.end_time = time.time()
            self.datastore.update_pipeline_run_status(self.run_id, 'failed')
            raise
    
    def _get_next_stage(self, current_stage: Stage, stage_levels: List[List[Stage]], current_level: int) -> Optional[Stage]:
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
        self,
        stage: Stage,
        next_stage: Stage,
        df: pd.DataFrame
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
                filtered_count += (start_chunk_count - len(filtered_chunk))
                
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
            filtered_count=0
        )
        
        self.stage_results[next_stage.name] = StageResult(
            stage_name=next_stage.name,
            input_count=len(df),
            output_count=len(df_out),
            filtered_count=filtered_count
        )
        
        if self.verbose:
            print(f"  {stage.name}: processed {len(df)} sequences")
            print(f"  {next_stage.name}: {len(df_out)} passed filter (filtered {filtered_count})")
        
        return df_out
    
    def run(self, enable_streaming: bool = False, **kwargs) -> Dict[str, StageResult]:
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
    
    def _get_config(self) -> Dict:
        """Get pipeline configuration for serialization."""
        return {
            'pipeline_type': self.pipeline_type,
            'run_id': self.run_id,
            'stages': [
                {
                    'name': s.name,
                    'type': s.__class__.__name__,
                    'depends_on': s.depends_on
                }
                for s in self.stages
            ]
        }
    
    def get_final_data(self) -> pd.DataFrame:
        """Get the final output DataFrame."""
        if not self._stage_data:
            raise RuntimeError("Pipeline has not been run yet")
        
        # Return data from the last stage
        last_stage_name = self.stages[-1].name
        return self._stage_data.get(last_stage_name, pd.DataFrame())
    
    def export_to_csv(self, output_path: Optional[Union[str, Path]] = None):
        """Export final results to CSV."""
        if output_path is None:
            output_path = self.output_dir / f'{self.run_id}_final.csv'
        
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
        for stage_name, result in self.stage_results.items():
            summary_data.append({
                'Stage': result.stage_name,
                'Input': result.input_count,
                'Output': result.output_count,
                'Filtered': result.filtered_count,
                'Cached': result.cached_count,
                'Computed': result.computed_count,
                'Time (s)': f"{result.elapsed_time:.1f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def __repr__(self):
        return (
            f"{self.pipeline_type}("
            f"run_id='{self.run_id}', "
            f"stages={len(self.stages)}, "
            f"status='{self.status}')"
        )
