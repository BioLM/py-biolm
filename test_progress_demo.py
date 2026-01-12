"""Demo script to test progress bars with ESMFold and peptides API."""

import asyncio
import os
import random
import sys
from biolmai.client import BioLMApiClient

# Note: For best results, run this script with:
#   python -u test_progress_demo.py
# or use the shell script:
#   ./test_progress_demo.sh
# This ensures unbuffered output so tqdm progress bars display correctly.

# Set your token if not already set
TOKEN = os.getenv("BIOLMAI_TOKEN") or os.getenv("KNOX_TOKEN") or os.getenv("TOKEN") or "b10a1c0deafaceb00c0ffeebada55a11fe55d15ea5eacc0ab1efacefeedc0de5"


def random_sequence(length=10):
    """Generate a random protein sequence."""
    return ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=length))


# Multiple sequences for ESMFold structure prediction
ESMFOLD_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK",
    "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH",
    "MDNELE",
    "MENDEL",
    "ISOTYPE",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVK",
    "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH",
    "MDNELE",
    "MENDEL",
    "ISOTYPE",
]


async def test_esmfold_multiple_structures():
    """Test progress bars with multiple ESMFold structure predictions."""
    print("=" * 60)
    print("Test 1: ESMFold - Multiple Structure Predictions")
    print("=" * 60)
    
    client = BioLMApiClient(
        "esmfold",
        api_key=TOKEN,
        progress=True,  # Enable progress bars (default)
        telemetry=True,  # Required for progress bars
    )
    
    try:
        # Predict structures for multiple sequences concurrently
        items = [{"sequence": seq} for seq in ESMFOLD_SEQUENCES[:5]]
        result = await client.predict(items=items, params={})
        print(f"\n✓ Success! Got {len(result)} structure(s)")
        # Small delay to keep progress bars visible
        await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        await client.shutdown()


async def test_peptides_large_batch():
    """Test progress bars with a large batch of peptides requests (100 sequences)."""
    print("\n" + "=" * 60)
    print("Test 2: Peptides - Large Batch (100 sequences)")
    print("=" * 60)
    
    client = BioLMApiClient(
        "peptides",
        api_key=TOKEN,
        progress=True,
        telemetry=True,
    )
    
    try:
        # Generate 100 random peptide sequences
        sequences = [random_sequence(length=random.randint(8, 15)) for _ in range(100)]
        items = [{"sequence": seq} for seq in sequences]
        
        # Peptides uses "encode" action, not "predict"
        result = await client.encode(items=items, params={})
        print(f"\n✓ Success! Got {len(result)} result(s) from 100 sequences")
        # Small delay to keep progress bars visible
        await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        await client.shutdown()


async def test_esmfold_concurrent_requests():
    """Test progress bars with multiple concurrent ESMFold requests."""
    print("\n" + "=" * 60)
    print("Test 3: ESMFold - Concurrent Requests (Same Algorithm)")
    print("=" * 60)
    
    client = BioLMApiClient(
        "esmfold",
        api_key=TOKEN,
        progress=True,
        telemetry=True,
    )
    
    try:
        # Create multiple concurrent requests - they should aggregate into one progress bar
        tasks = [
            client.predict(items=[{"sequence": ESMFOLD_SEQUENCES[0]}], params={}),
            client.predict(items=[{"sequence": ESMFOLD_SEQUENCES[1]}], params={}),
            client.predict(items=[{"sequence": ESMFOLD_SEQUENCES[2]}], params={}),
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"\n✓ Success! Got {len(results)} results from {len(tasks)} concurrent requests")
        # Small delay to keep progress bars visible
        await asyncio.sleep(0.5)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        await client.shutdown()


async def main():
    """Run all tests."""
    print("\n🧪 Testing Progress Bars with ESMFold and Peptides\n")
    
    # Test 1: ESMFold multiple structures
    await test_esmfold_multiple_structures()
    
    # Test 2: Peptides large batch (100 sequences)
    await test_peptides_large_batch()
    
    # Test 3: ESMFold concurrent requests
    await test_esmfold_concurrent_requests()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

