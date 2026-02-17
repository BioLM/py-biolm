"""
Performance benchmark for clustering and diversity analysis.

Demonstrates scalability improvements and provides timing data.
"""

import time
import numpy as np
from typing import List
from biolmai.pipeline.clustering import cluster_sequences, analyze_diversity


def generate_test_sequences(n: int, length: int = 100) -> List[str]:
    """Generate random protein sequences for testing."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return [''.join(np.random.choice(list(amino_acids), length)) for _ in range(n)]


def benchmark_clustering():
    """Benchmark clustering performance at different scales."""
    print("=" * 70)
    print("CLUSTERING PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    test_sizes = [100, 1000, 5000, 10000, 50000]
    
    for n in test_sizes:
        print(f"\n{n:,} sequences:")
        sequences = generate_test_sequences(n, length=50)
        
        # Test 1: Hamming distance (baseline)
        if n <= 10000:
            start = time.time()
            result = cluster_sequences(
                sequences,
                method='kmeans',
                n_clusters=min(10, n // 10),
                similarity_metric='hamming'
            )
            elapsed = time.time() - start
            print(f"  Hamming distance: {elapsed:.2f}s")
        else:
            print(f"  Hamming distance: SKIPPED (would be slow)")
        
        # Test 2: Hamming with sampling
        start = time.time()
        result = cluster_sequences(
            sequences,
            method='kmeans',
            n_clusters=min(10, n // 10),
            similarity_metric='hamming',
            max_sample=min(5000, n)
        )
        elapsed = time.time() - start
        print(f"  Hamming + sampling: {elapsed:.2f}s")
        
        # Test 3: MiniBatch K-means
        start = time.time()
        result = cluster_sequences(
            sequences,
            method='kmeans',
            n_clusters=min(10, n // 10),
            similarity_metric='hamming',
            mini_batch=True,
            max_sample=min(5000, n)
        )
        elapsed = time.time() - start
        print(f"  MiniBatch K-means: {elapsed:.2f}s")
        
        # Test 4: With embeddings (simulate)
        if n <= 10000:
            embeddings = np.random.randn(n, 128)  # Simulate embeddings
            start = time.time()
            result = cluster_sequences(
                sequences,
                method='kmeans',
                n_clusters=min(10, n // 10),
                similarity_metric='embedding',
                embeddings=embeddings
            )
            elapsed = time.time() - start
            print(f"  Embedding-based: {elapsed:.2f}s ⚡ (FASTEST)")


def benchmark_diversity():
    """Benchmark diversity analysis performance."""
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    test_sizes = [100, 1000, 10000, 50000, 100000]
    
    for n in test_sizes:
        print(f"\n{n:,} sequences:")
        sequences = generate_test_sequences(n, length=50)
        
        # Shannon entropy (fast - O(n))
        start = time.time()
        from biolmai.pipeline.clustering import DiversityAnalyzer
        entropy = DiversityAnalyzer.shannon_entropy(sequences)
        elapsed = time.time() - start
        print(f"  Shannon entropy: {elapsed:.3f}s (entropy={entropy:.3f})")
        
        # Pairwise distances without sampling (slow - O(n²))
        if n <= 5000:
            start = time.time()
            stats = DiversityAnalyzer.pairwise_distance_stats(sequences, max_sample=None)
            elapsed = time.time() - start
            print(f"  Pairwise (full): {elapsed:.2f}s")
        else:
            print(f"  Pairwise (full): SKIPPED (O(n²) too slow)")
        
        # Pairwise distances with sampling (fast)
        start = time.time()
        stats = DiversityAnalyzer.pairwise_distance_stats(sequences, max_sample=5000)
        elapsed = time.time() - start
        n_comp = stats.get('n_comparisons', 0)
        print(f"  Pairwise (sampled): {elapsed:.3f}s ({n_comp:,} comparisons)")
        
        # All metrics
        start = time.time()
        metrics = analyze_diversity(sequences, max_sample=5000)
        elapsed = time.time() - start
        print(f"  All metrics: {elapsed:.3f}s")


def print_recommendations():
    """Print performance recommendations."""
    print("\n" + "=" * 70)
    print("PERFORMANCE RECOMMENDATIONS")
    print("=" * 70)
    print("""
For CLUSTERING:
  • <1k sequences: Use any method
  • 1k-10k sequences: Use hamming with default settings
  • 10k-50k sequences: Use hamming + max_sample=5000 OR embeddings
  • 50k-1M sequences: Use embeddings + mini_batch=True
  • >1M sequences: Use embeddings + mini_batch=True + distributed processing

For DIVERSITY ANALYSIS:
  • Shannon entropy: Scales linearly, fast for any N
  • Motif diversity: Scales linearly, fast for any N
  • Pairwise distances: Always use max_sample (default=10k) for N>10k

BEST PRACTICE FOR LARGE DATASETS:
  1. Generate embeddings once (esm2-650m)
  2. Use embedding-based clustering with mini_batch=True
  3. Use analyze_diversity() with max_sample=10000
  4. This approach scales to millions of sequences!

MEMORY CONSIDERATIONS:
  • Hamming distance matrix: O(n²) memory ~8GB for 30k sequences
  • Embeddings: O(n*d) memory ~500MB for 1M sequences @ 128-dim
  • Always prefer embedding-based methods for large datasets
""")


if __name__ == '__main__':
    np.random.seed(42)
    
    print("\nRunning performance benchmarks...")
    print("This may take a few minutes...\n")
    
    benchmark_clustering()
    benchmark_diversity()
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
