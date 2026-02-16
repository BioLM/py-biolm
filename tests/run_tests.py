#!/usr/bin/env python
"""
Test runner for BioLM Pipeline tests.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(pattern='test_*.py', verbosity=2):
    """
    Run all tests matching the pattern.
    
    Args:
        pattern: Test file pattern (default: 'test_*.py')
        verbosity: Verbosity level (default: 2)
    
    Returns:
        True if all tests passed, False otherwise
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(str(start_dir), pattern=pattern)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*60)
    
    return result.wasSuccessful()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BioLM Pipeline tests')
    parser.add_argument(
        '-p', '--pattern',
        default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet output'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Run tests
    success = run_tests(pattern=args.pattern, verbosity=verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
