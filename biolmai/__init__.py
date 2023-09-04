"""Top-level package for BioLM AI."""
__author__ = """Nikhil Haas"""
__email__ = 'nikhil@biolm.ai'
__version__ = '0.1.0'

from biolmai.biolmai import get_api_token, api_call

__all__ = [
    "get_api_token",
    "api_call",
]
