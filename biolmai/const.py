import os

if os.environ.get('BIOLMAI_LOCAL', False):
    # For local development and tests only
    BASE_DOMAIN = 'http://localhost:8000'
else:
    BASE_DOMAIN = 'https://biolm.ai'

USER_BIOLM_DIR = os.path.join(os.path.expanduser('~'), '.biolmai')
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, 'credentials')
GEN_TOKEN_URL = f'{BASE_DOMAIN}/ui/accounts/user-api-tokens/'
MULTIPROCESS_THREADS = os.environ.get('BIOLMAI_THREADS', False)
