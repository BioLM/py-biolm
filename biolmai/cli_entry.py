#!/usr/bin/env python
"""Entry point wrapper to suppress urllib3 warnings."""
import sys
import os
import warnings

# CRITICAL: Redirect stderr file descriptor BEFORE any other imports
# urllib3 prints warnings directly to fd 2, so we must redirect it early
_devnull = open(os.devnull, 'w')
_original_stderr_fd = os.dup(2)  # Save original stderr file descriptor
os.dup2(_devnull.fileno(), 2)  # Redirect stderr (fd 2) to /dev/null
sys.stderr = _devnull

# Suppress all warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

# Patch showwarning to filter urllib3 warnings
_original_showwarning = warnings.showwarning
def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    msg_str = str(message) if message else ''
    filename_str = str(filename) if filename else ''
    if 'urllib3' in filename_str or 'urllib3' in msg_str or 'OpenSSL' in msg_str:
        return
    _original_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _filtered_showwarning

# Try to import urllib3 early and disable warnings
try:
    import urllib3
    urllib3.disable_warnings()
except (ImportError, AttributeError):
    pass

# Import CLI (urllib3 will be imported via httpx)
from biolmai.cli import cli

# Restore stderr after all imports
os.dup2(_original_stderr_fd, 2)  # Restore original stderr file descriptor
sys.stderr = sys.__stderr__  # Restore sys.stderr
_devnull.close()
os.close(_original_stderr_fd)

if __name__ == '__main__':
    sys.exit(cli())
