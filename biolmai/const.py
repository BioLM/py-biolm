import multiprocessing
import os

cpu_count = multiprocessing.cpu_count()
max_threads = cpu_count * 4

# Check for BASE_DOMAIN environment variable first (allows override)
if os.environ.get("BASE_DOMAIN"):
    BASE_DOMAIN = os.environ.get("BASE_DOMAIN")
    # Ensure it has a scheme
    if not BASE_DOMAIN.startswith(("http://", "https://")):
        BASE_DOMAIN = f"http://{BASE_DOMAIN}"
elif os.environ.get("BIOLMAI_LOCAL", False):
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

# OAuth 2.0 configuration
BIOLMAI_PUBLIC_CLIENT_ID = os.environ.get("BIOLMAI_OAUTH_CLIENT_ID", "")
# Check both CLIENT_SECRET and BIOLMAI_OAUTH_CLIENT_SECRET for compatibility
BIOLMAI_OAUTH_CLIENT_SECRET = os.environ.get("BIOLMAI_OAUTH_CLIENT_SECRET") or os.environ.get("CLIENT_SECRET", "")
OAUTH_AUTHORIZE_URL = f"{BASE_DOMAIN}/o/authorize/"
OAUTH_TOKEN_URL = f"{BASE_DOMAIN}/o/token/"
# For introspection, use backend URL (8000) if BASE_DOMAIN points to frontend (7777)
# This is because OAuth endpoints are on the backend, not the frontend proxy
if BASE_DOMAIN == "http://localhost:7777" or BASE_DOMAIN.endswith(":7777"):
    OAUTH_INTROSPECT_URL = "http://localhost:8000/o/introspect/"
else:
    OAUTH_INTROSPECT_URL = f"{BASE_DOMAIN}/o/introspect/"
OAUTH_REDIRECT_URI = "http://localhost:8765/callback"
