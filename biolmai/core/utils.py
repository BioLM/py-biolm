"""Utility functions for BioLM SDK."""

from itertools import islice


def is_list_of_lists(items, check_n=10):
    # Accepts any iterable, checks first N items for list/tuple-ness
    # Returns (is_list_of_lists, first_n_items, rest_iter)
    if isinstance(items, (list, tuple)):
        if not items:
            return False, [], iter(())
        first_n = items[:check_n]
        is_lol = all(isinstance(x, (list, tuple)) for x in first_n)
        return is_lol, first_n, iter(items[check_n:])
    # For iterators/generators: consume first N, return rest (no tee - tee would
    # duplicate first N when caller chains first_n + rest)
    first_n = list(islice(items, check_n))
    is_lol = all(isinstance(x, (list, tuple)) for x in first_n) and bool(first_n)
    return is_lol, first_n, items


def batch_iterable(iterable, batch_size):
    # Yields lists of up to batch_size from any iterable, deleting as we go
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
