import multiprocessing
import os

cpu_count = multiprocessing.cpu_count()
max_threads = cpu_count * 4

if os.environ.get("BIOLMAI_LOCAL", False):
    # For local development and tests only
    BASE_DOMAIN = "http://localhost:8000"
else:
    BASE_DOMAIN = "https://biolm.ai"

USER_BIOLM_DIR = os.path.join(os.path.expanduser("~"), ".biolmai")
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, "credentials")
GEN_TOKEN_URL = f"{BASE_DOMAIN}/ui/accounts/user-api-tokens/"
MULTIPROCESS_THREADS = os.environ.get("BIOLMAI_THREADS", 1)
if isinstance(MULTIPROCESS_THREADS, str) and not MULTIPROCESS_THREADS:
    MULTIPROCESS_THREADS = 1
if int(MULTIPROCESS_THREADS) > max_threads or int(MULTIPROCESS_THREADS) > 128:
    err = (
        f"Maximum threads allowed is 4x number of CPU cores ("
        f"{max_threads}) or 128, whichever is lower."
    )
    err += " Please update environment variable BIOLMAI_THREADS."
    raise ValueError(err)
elif int(MULTIPROCESS_THREADS) <= 0:
    err = "Environment variable BIOLMAI_THREADS must be a positive integer."
    raise ValueError(err)
BASE_API_URL_V1 = f"{BASE_DOMAIN}/api/v1"
BASE_API_URL = f"{BASE_DOMAIN}/api/v2"
