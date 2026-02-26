"""
Masked Language Model (MLM) remasking utilities.

Provides functionality for iterative remasking and prediction using
masked language models like ESM, ESM-1v, ESM-2.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class RemaskingConfig:
    """
    Configuration for MLM remasking.

    Args:
        model_name: MLM model to use (e.g., 'esm-150m', 'esm-650m', 'esm-3b', 'esm3', 'esmc')
        mask_fraction: Fraction of positions to mask (default: 0.15)
        mask_positions: Specific positions to mask, or 'auto' for random
        num_iterations: Number of remasking iterations (default: 1)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling (default: None)
        top_p: Nucleus sampling (default: None)
        mask_token: Token to use for masking (default: '<mask>')
        conserved_positions: Positions that should never be masked
        mask_strategy: Strategy for selecting positions ('random', 'low_confidence', 'blocks')
        block_size: Size of blocks for block masking (default: 3)
        confidence_threshold: Threshold for low-confidence masking (default: 0.8)

    Example:
        >>> # ESM2 150M remasking
        >>> config = RemaskingConfig(
        ...     model_name='esm-150m',
        ...     mask_fraction=0.15,
        ...     num_iterations=5,
        ...     temperature=1.0
        ... )
        >>>
        >>> # ESM3 with higher temperature
        >>> config = RemaskingConfig(
        ...     model_name='esm3',
        ...     mask_fraction=0.20,
        ...     temperature=1.5
        ... )
    """

    model_name: str = "esm-150m"  # Default to ESM2 150M
    mask_fraction: float = 0.15
    mask_positions: Union[str, List[int]] = "auto"
    num_iterations: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    mask_token: str = "<mask>"
    conserved_positions: Optional[List[int]] = None
    mask_strategy: str = "random"
    block_size: int = 3
    confidence_threshold: float = 0.8


class MLMRemasker:
    """
    Masked Language Model remasking utility.

    Handles iterative masking and prediction for generating sequence variants
    using masked language models.

    Args:
        config: RemaskingConfig instance
        api_client: BioLM API client (optional, for actual predictions)
        model_name: Model name (e.g., 'esm2', 'esm1v')

    Example:
        >>> config = RemaskingConfig(mask_fraction=0.15, num_iterations=5)
        >>> remasker = MLMRemasker(config, model_name='esm2')
        >>> variants = remasker.generate_variants('MKTAYIAKQRQ', num_variants=10)
    """

    def __init__(
        self, config: RemaskingConfig, api_client=None, model_name: Optional[str] = None
    ):
        self.config = config
        self.api_client = api_client
        # Use model_name from config, or override if provided
        self.model_name = model_name or config.model_name
        self.random = random.Random(42)  # For reproducibility

    def select_mask_positions(
        self, sequence: str, confidences: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select positions to mask based on strategy.

        Args:
            sequence: Input sequence
            confidences: Optional confidence scores per position (for low_confidence strategy)

        Returns:
            List of positions to mask (0-indexed)
        """
        seq_len = len(sequence)

        # Explicit positions
        if isinstance(self.config.mask_positions, list):
            return [p for p in self.config.mask_positions if 0 <= p < seq_len]

        # Calculate number to mask
        num_to_mask = max(1, int(seq_len * self.config.mask_fraction))

        # Get conserved positions
        conserved = set(self.config.conserved_positions or [])
        available_positions = [i for i in range(seq_len) if i not in conserved]

        if len(available_positions) < num_to_mask:
            num_to_mask = len(available_positions)

        # Select based on strategy
        if self.config.mask_strategy == "random":
            return self.random.sample(available_positions, num_to_mask)

        elif self.config.mask_strategy == "low_confidence":
            if confidences is None:
                # Fallback to random
                return self.random.sample(available_positions, num_to_mask)

            # Select positions with lowest confidence
            available_confidences = [(i, confidences[i]) for i in available_positions]
            available_confidences.sort(key=lambda x: x[1])

            return [i for i, _ in available_confidences[:num_to_mask]]

        elif self.config.mask_strategy == "blocks":
            # Mask contiguous blocks
            positions = []
            block_size = self.config.block_size

            while len(positions) < num_to_mask:
                # Pick random start position
                start = self.random.choice(available_positions)

                # Add block
                for i in range(start, min(start + block_size, seq_len)):
                    if (
                        i not in conserved
                        and i not in positions
                        and i in available_positions
                    ):
                        positions.append(i)
                        if len(positions) >= num_to_mask:
                            break

            return positions[:num_to_mask]

        else:
            raise ValueError(f"Unknown mask_strategy: {self.config.mask_strategy}")

    def create_masked_sequence(self, sequence: str, positions: List[int]) -> str:
        """
        Create masked sequence with mask token at specified positions.

        Args:
            sequence: Original sequence
            positions: Positions to mask

        Returns:
            Masked sequence
        """
        seq_list = list(sequence)

        for pos in positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = self.config.mask_token

        return "".join(seq_list)

    async def predict_masked_positions(
        self, sequence: str, mask_positions: List[int]
    ) -> Tuple[str, Dict[int, float]]:
        """
        Predict amino acids at masked positions.

        Args:
            sequence: Original (unmasked) sequence — the API receives the clean
                sequence and mask_positions as a parameter; it never sees mask tokens.
            mask_positions: 0-indexed positions to replace.

        Returns:
            Tuple of (predicted_sequence, confidences_dict)
        """
        if self.api_client is None:
            # Mock: replace positions in-place with random amino acids
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            confidences = {}
            seq_list = list(sequence)
            for pos in mask_positions:
                replacement = self.random.choice(amino_acids)
                seq_list[pos] = replacement
                confidences[pos] = self.random.uniform(0.5, 1.0)
            return "".join(seq_list), confidences

        else:
            # Real API: send the original sequence + mask_positions as params.
            # The API internally masks and infills; it rejects sequences that
            # contain non-amino-acid characters such as '<mask>'.
            result = await self.api_client.predict(
                items=[{"sequence": sequence}],
                params={
                    "mask_positions": mask_positions,
                    "temperature": self.config.temperature,
                    "top_k": self.config.top_k,
                    "top_p": self.config.top_p,
                },
            )

            # Extract predicted sequence and confidences
            if isinstance(result, list) and len(result) > 0:
                pred_result = result[0]

                if isinstance(pred_result, dict):
                    predicted_seq = pred_result.get("sequence")
                    confidences = pred_result.get("confidences", {})

                    if predicted_seq is None:
                        raise ValueError(
                            f"API result missing 'sequence' field. Got: {pred_result}"
                        )

                    return predicted_seq, confidences
                else:
                    # Result is not a dict, try to use it as a sequence
                    predicted_seq = str(pred_result)
                    if self.config.mask_token in predicted_seq:
                        raise ValueError(
                            f"API returned sequence still contains mask token: {predicted_seq}"
                        )
                    return predicted_seq, {}

            else:
                raise ValueError(
                    f"API returned empty or invalid result: {result}. "
                    f"Expected list with at least one element."
                )

    async def generate_variant(
        self, parent_sequence: str, iteration: int = 0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a single variant through remasking (async).

        Args:
            parent_sequence: Starting sequence
            iteration: Iteration number (for seeding)

        Returns:
            Tuple of (variant_sequence, metadata_dict)
        """
        current_sequence = parent_sequence
        all_mask_positions = []
        all_confidences = {}

        # Perform remasking iterations
        for iter_num in range(self.config.num_iterations):
            # Select positions to mask
            mask_positions = self.select_mask_positions(current_sequence)

            # Predict: pass original sequence + positions; API handles masking internally
            predicted_seq, confidences = await self.predict_masked_positions(
                current_sequence, mask_positions
            )

            # Update
            current_sequence = predicted_seq
            all_mask_positions.extend(mask_positions)
            all_confidences.update(confidences)

        # Calculate statistics
        num_mutations = sum(
            1
            for i, (a, b) in enumerate(zip(parent_sequence, current_sequence))
            if a != b
        )

        mutation_rate = (
            num_mutations / len(parent_sequence) if len(parent_sequence) > 0 else 0
        )

        metadata = {
            "parent_sequence": parent_sequence,
            "num_iterations": self.config.num_iterations,
            "mask_positions": list(set(all_mask_positions)),
            "num_mutations": num_mutations,
            "mutation_rate": mutation_rate,
            "avg_confidence": (
                np.mean(list(all_confidences.values())) if all_confidences else 0.0
            ),
            "mask_strategy": self.config.mask_strategy,
            "mask_fraction": self.config.mask_fraction,
        }

        return current_sequence, metadata

    async def generate_variants(
        self, parent_sequence: str, num_variants: int = 100, deduplicate: bool = True
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate multiple variants through remasking (async).

        Args:
            parent_sequence: Starting sequence
            num_variants: Number of variants to generate
            deduplicate: Remove duplicate sequences

        Returns:
            List of (variant_sequence, metadata) tuples
        """
        variants = []
        seen_sequences = {parent_sequence} if deduplicate else set()

        attempts = 0
        max_attempts = num_variants * 10  # Avoid infinite loop

        while len(variants) < num_variants and attempts < max_attempts:
            attempts += 1

            # generate_variant is async — must be awaited
            variant_seq, metadata = await self.generate_variant(
                parent_sequence, iteration=attempts
            )

            if deduplicate:
                if variant_seq in seen_sequences:
                    continue
                seen_sequences.add(variant_seq)

            variants.append((variant_seq, metadata))

        return variants

    async def iterative_refinement(
        self,
        sequence: str,
        fitness_function: callable,
        num_iterations: int = 10,
        population_size: int = 20,
        keep_top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform iterative refinement using remasking and a fitness function (async).

        Args:
            sequence: Starting sequence
            fitness_function: Function that scores sequences (higher is better)
            num_iterations: Number of refinement iterations
            population_size: Number of variants per iteration
            keep_top_k: Number of top sequences to keep per iteration

        Returns:
            List of (sequence, fitness, metadata) tuples for final population
        """
        current_population = [(sequence, fitness_function(sequence), {})]

        for iteration in range(num_iterations):
            new_variants = []

            for parent_seq, parent_fitness, _ in current_population:
                variants = await self.generate_variants(
                    parent_seq,
                    num_variants=population_size // len(current_population),
                    deduplicate=True,
                )

                for variant_seq, metadata in variants:
                    fitness = fitness_function(variant_seq)
                    new_variants.append((variant_seq, fitness, metadata))

            all_sequences = current_population + new_variants
            all_sequences.sort(key=lambda x: x[1], reverse=True)
            current_population = all_sequences[:keep_top_k]

            if current_population:
                print(
                    f"Iteration {iteration + 1}: Best fitness = {current_population[0][1]:.3f}"
                )

        return current_population


def create_remasker_from_dict(config_dict: Dict[str, Any], **kwargs) -> MLMRemasker:
    """
    Create MLMRemasker from a configuration dictionary.

    Args:
        config_dict: Dictionary with configuration parameters
        **kwargs: Additional arguments for MLMRemasker

    Returns:
        MLMRemasker instance

    Example:
        >>> config = {
        ...     'mask_fraction': 0.2,
        ...     'num_iterations': 5,
        ...     'temperature': 1.0
        ... }
        >>> remasker = create_remasker_from_dict(config, model_name='esm2')
    """
    config = RemaskingConfig(**config_dict)
    return MLMRemasker(config, **kwargs)


# Predefined configurations for common use cases

CONSERVATIVE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.10,
    num_iterations=3,
    temperature=0.5,
    mask_strategy="random",
)

MODERATE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.15,
    num_iterations=5,
    temperature=1.0,
    mask_strategy="random",
)

AGGRESSIVE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.25,
    num_iterations=10,
    temperature=1.5,
    mask_strategy="random",
)

BLOCK_MASKING_CONFIG = RemaskingConfig(
    mask_fraction=0.15,
    num_iterations=5,
    temperature=1.0,
    mask_strategy="blocks",
    block_size=3,
)
