import json
import os
import pprint
import stat

import click
import requests

from biolmai.const import ACCESS_TOK_PATH, BASE_DOMAIN, GEN_TOKEN_URL, \
    USER_BIOLM_DIR


def validate_user_auth(api_token=None, access=None, refresh=None):
    """Validates an API token, to be used as 'Authorization: Token 1235abc'
    authentication method."""
    url = f'{BASE_DOMAIN}/api/v1/auth/login-check/'
    if api_token is not None:
        headers = {'Authorization': f'Token {api_token}'}
    else:
        headers = {
            'Cookie': 'access={};refresh={}'.format(access, refresh),
            'Content-Type': 'application/json'
        }
    try:
        # click.echo(headers)
        r = requests.post(url=url, headers=headers)
        # click.echo('Printing')
        # click.echo(r.text)
        json_response = r.json()
        pretty_json = pprint.pformat(json_response, indent=2)
        click.echo(pretty_json)
    except Exception:
        click.echo("Token validation failed!\n")
        raise
    else:
        return r


def refresh_access_token(refresh):
    """Attempt to refresh temporary user access token, by using their refresh
    token, which has a longer TTL."""
    url = f'{BASE_DOMAIN}/api/auth/token/refresh/'
    headers = {
        'Cookie': 'refresh={}'.format(refresh),
        'Content-Type': 'application/json'
    }
    r = requests.post(url=url, headers=headers)
    json_response = r.json()
    if r.status_code != 200:
        pretty_json = pprint.pformat(json_response, indent=2)
        click.echo(pretty_json)
        click.echo(f"Token refresh failed! Please login by "
                   "running `biolmai login`.\n")
    else:
        click.echo("User access token successfully refreshed. Saving credentials...")
        access_refresh_dict = {'access': json_response['access'],
                               'refresh': refresh}
        save_access_refresh_token(access_refresh_dict)


def get_auth_status():
    environ_token = os.environ.get('BIOLMAI_TOKEN', None)
    if environ_token:
        msg = "Environment variable BIOLMAI_TOKEN detected. Validating token..."
        click.echo(msg)
        validate_user_auth(api_token=environ_token)
    elif os.path.exists(ACCESS_TOK_PATH):
        msg = f"Credentials file found {ACCESS_TOK_PATH}. Validating token..."
        click.echo(msg)
        with open(ACCESS_TOK_PATH, 'r') as f:
            access_refresh_dict = json.load(f)
        access = access_refresh_dict.get('access')
        refresh = access_refresh_dict.get('refresh')
        resp = validate_user_auth(access=access, refresh=refresh)
        if resp.status_code != 200:
            click.echo("Access token validation failed. Attempting to refresh token...")
            # Attempt to use the 'refresh' token to get a new 'access' token
            refresh_access_token(refresh)
    else:
        msg = f"No https://biolm.ai login credentials found. Please " \
              f"set the environment variable BIOLMAI_TOKEN to a token from {GEN_TOKEN_URL}, or login by " \
              "running `biolmai login`."
        click.echo(msg)


def generate_access_token(uname, password):
    """Generate a TTL-expiry access and refresh token, to be used as
    'Cookie: acccess=; refresh=;" headers, or the access token only as a
    'Authorization: Bearer 1235abc' token.

    The refresh token will expire in hours or days, while the access token
    will have a shorter TTL, more like hours. Meaning, this method will
    require periodically re-logging in, due to the token expiration time. For a
    more permanent auth method for the API, use an API token by setting the
    BIOLMAI_TOKEN environment variable.
    """
    url = f'{BASE_DOMAIN}/api/auth/token/'
    try:
        r = requests.post(url=url, data={'username': uname, 'password': password})
        json_response = r.json()
    except Exception:
        click.echo("Login failed!\n")
        raise
    if r.status_code != 200:
        click.echo("Login failed!\n")
        resp_json = r.json()
        pretty_json = pprint.pformat(resp_json, indent=2)
        click.echo(pretty_json)
        return {}
    else:
        click.echo("Login succeeded!\n")
        return json_response


def save_access_refresh_token(access_refresh_dict):
    """Save temporary access and refresh tokens to user folder for future
    use."""
    os.makedirs(USER_BIOLM_DIR, exist_ok=True)
    # Save token
    with open(ACCESS_TOK_PATH, 'w') as f:
        json.dump(access_refresh_dict, f)
    os.chmod(ACCESS_TOK_PATH, stat.S_IRUSR | stat.S_IWUSR)
    # Validate token and print user info
    # click.echo(access_refresh_dict)
    access = access_refresh_dict.get('access')
    refresh = access_refresh_dict.get('refresh')
    validate_user_auth(access=access, refresh=refresh)
