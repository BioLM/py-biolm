"""Utility functions for BioLM SDK."""
from itertools import tee, islice
from typing import Tuple, Any, Iterable, List


def is_list_of_lists(items, check_n=10):
    """Check if items is a list of lists.
    
    Accepts any iterable, checks first N items for list/tuple-ness.
    Returns (is_list_of_lists, first_n_items, rest_iter).
    """
    if isinstance(items, (list, tuple)):
        if not items:
            return False, [], iter(())
        first_n = items[:check_n]
        is_lol = all(isinstance(x, (list, tuple)) for x in first_n)
        return is_lol, first_n, iter(items[check_n:])
    # For iterators/generators
    items, items_copy = tee(items)
    first_n = list(islice(items_copy, check_n))
    is_lol = all(isinstance(x, (list, tuple)) for x in first_n) and bool(first_n)
    return is_lol, first_n, items


def batch_iterable(iterable, batch_size):
    """Yield lists of up to batch_size from any iterable.
    
    Args:
        iterable: Any iterable to batch
        batch_size: Maximum size of each batch
        
    Yields:
        Lists of items from the iterable, each up to batch_size in length
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

