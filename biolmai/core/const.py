import multiprocessing
import os

cpu_count = multiprocessing.cpu_count()
max_threads = cpu_count * 4

# Check for BIOLMAI_BASE_DOMAIN environment variable first (allows override)
if os.environ.get("BIOLMAI_BASE_DOMAIN"):
    BIOLMAI_BASE_DOMAIN = os.environ.get("BIOLMAI_BASE_DOMAIN")
    # Ensure it has a scheme
    if not BIOLMAI_BASE_DOMAIN.startswith(("http://", "https://")):
        BIOLMAI_BASE_DOMAIN = f"http://{BIOLMAI_BASE_DOMAIN}"
elif str(os.environ.get("BIOLMAI_LOCAL", False)).lower() == "true":
    # For local development and tests only
    BIOLMAI_BASE_DOMAIN = "http://localhost:8000"
else:
    BIOLMAI_BASE_DOMAIN = "https://biolm.ai"

USER_BIOLM_DIR = os.path.join(os.path.expanduser("~"), ".biolmai")
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, "credentials")
GEN_TOKEN_URL = f"{BIOLMAI_BASE_DOMAIN}/ui/accounts/user-api-tokens/"
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
BASE_API_URL_V1 = f"{BIOLMAI_BASE_DOMAIN}/api/v1"
BASE_API_URL = f"{BIOLMAI_BASE_DOMAIN}/api/v3"

# Default base URL for new client classes (BioLMApiClient, BioLMApi, biolm)
# Priority: BIOLMAI_BASE_API_URL env var > BIOLMAI_BASE_DOMAIN env var > default
if os.environ.get("BIOLMAI_BASE_API_URL"):
    # Highest priority: use BIOLMAI_BASE_API_URL if set
    BIOLMAI_BASE_API_URL = os.environ.get("BIOLMAI_BASE_API_URL")
elif os.environ.get("BIOLMAI_BASE_DOMAIN"):
    # Second priority: construct from BIOLMAI_BASE_DOMAIN
    domain = os.environ.get("BIOLMAI_BASE_DOMAIN")
    # Ensure it has a scheme
    if not domain.startswith(("http://", "https://")):
        domain = f"http://{domain}"
    BIOLMAI_BASE_API_URL = f"{domain}/api/v3"
else:
    # Default
    BIOLMAI_BASE_API_URL = "https://biolm.ai/api/v3"

# OAuth 2.0 configuration
BIOLMAI_PUBLIC_CLIENT_ID = os.environ.get("BIOLMAI_OAUTH_CLIENT_ID", "2t_fFfnx9UjgmVp8EGbJRL24UbVynZ5Yo2JOv_R2eQc")
# Check both CLIENT_SECRET and BIOLMAI_OAUTH_CLIENT_SECRET for compatibility
BIOLMAI_OAUTH_CLIENT_SECRET = os.environ.get("BIOLMAI_OAUTH_CLIENT_SECRET") or os.environ.get("CLIENT_SECRET", "")
OAUTH_AUTHORIZE_URL = f"{BIOLMAI_BASE_DOMAIN}/o/authorize/"
OAUTH_TOKEN_URL = f"{BIOLMAI_BASE_DOMAIN}/o/token/"
# For introspection, use backend URL (8000) if BIOLMAI_BASE_DOMAIN points to frontend (7777)
# This is because OAuth endpoints are on the backend, not the frontend proxy
if BIOLMAI_BASE_DOMAIN == "http://localhost:7777" or BIOLMAI_BASE_DOMAIN.endswith(":7777"):
    OAUTH_INTROSPECT_URL = "http://localhost:8000/o/introspect/"
else:
    OAUTH_INTROSPECT_URL = f"{BIOLMAI_BASE_DOMAIN}/o/introspect/"
OAUTH_REDIRECT_URI = "http://localhost:8765/callback"
