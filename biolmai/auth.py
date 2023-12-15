import json
import os
import pprint
import stat

import click
import requests

from biolmai.const import ACCESS_TOK_PATH, BASE_DOMAIN, GEN_TOKEN_URL, USER_BIOLM_DIR


def validate_user_auth(api_token=None, access=None, refresh=None):
    """Validates an API token, to be used as 'Authorization: Token 1235abc'
    authentication method."""
    url = f"{BASE_DOMAIN}/api/v1/auth/login-check/"
    if api_token is not None:
        headers = {"Authorization": f"Token {api_token}"}
    else:
        headers = {
            "Cookie": f"access={access};refresh={refresh}",
            "Content-Type": "application/json",
        }
    try:
        r = requests.post(url=url, headers=headers)
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
    url = f"{BASE_DOMAIN}/api/auth/token/refresh/"
    headers = {"Cookie": f"refresh={refresh}", "Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers)
    json_response = r.json()
    if r.status_code != 200 or (r.status_code == 200 and "code" in r.json()):
        pretty_json = pprint.pformat(json_response, indent=2)
        click.echo(pretty_json)
        click.echo(
            "Token refresh failed! Please login by " "running `biolmai login`.\n"
        )
        return False
    else:
        access_refresh_dict = {"access": json_response["access"], "refresh": refresh}
        save_access_refresh_token(access_refresh_dict)
        return True


def get_auth_status():
    environ_token = os.environ.get("BIOLMAI_TOKEN", None)
    if environ_token:
        msg = "Environment variable BIOLMAI_TOKEN detected. Validating token..."
        click.echo(msg)
        validate_user_auth(api_token=environ_token)
    elif os.path.exists(ACCESS_TOK_PATH):
        msg = f"Credentials file found {ACCESS_TOK_PATH}. Validating token..."
        click.echo(msg)
        with open(ACCESS_TOK_PATH) as f:
            access_refresh_dict = json.load(f)
        access = access_refresh_dict.get("access")
        refresh = access_refresh_dict.get("refresh")
        resp = validate_user_auth(access=access, refresh=refresh)
        if resp.status_code != 200 or (
            resp.status_code == 200 and "code" in resp.json()
        ):
            click.echo("Access token validation failed. Attempting to refresh token...")
            # Attempt to use the 'refresh' token to get a new 'access' token
            if not refresh_access_token(refresh):
                click.echo("Unexpected refresh token error.")
            else:
                click.echo("Access token refresh was successful.")
    else:
        msg = (
            f"No https://biolm.ai credentials found. Please "
            f"set the environment variable BIOLMAI_TOKEN to a token from "
            f"{GEN_TOKEN_URL}, or login by running `biolmai login`."
        )
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
    url = f"{BASE_DOMAIN}/api/auth/token/"
    try:
        r = requests.post(url=url, data={"username": uname, "password": password})
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
    with open(ACCESS_TOK_PATH, "w") as f:
        json.dump(access_refresh_dict, f)
    os.chmod(ACCESS_TOK_PATH, stat.S_IRUSR | stat.S_IWUSR)
    # Validate token and print user info
    access = access_refresh_dict.get("access")
    refresh = access_refresh_dict.get("refresh")
    validate_user_auth(access=access, refresh=refresh)


def get_api_token():
    """Get a BioLM API token to use with future API requests.

    Copied from https://api.biolm.ai/#d7f87dfd-321f-45ae-99b6-eb203519ddeb.
    """
    url = "https://biolm.ai/api/auth/token/"

    payload = json.dumps(
        {
            "username": os.environ.get("BIOLM_USER"),
            "password": os.environ.get("BIOLM_PASSWORD"),
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()

    return response_json


def get_user_auth_header():
    """Returns a dict with the appropriate Authorization header, either using
    an API token from BIOLMAI_TOKEN environment variable, or by reading the
    credentials file at ~/.biolmai/credntials next."""
    api_token = os.environ.get("BIOLMAI_TOKEN", None)
    if api_token:
        headers = {"Authorization": f"Token {api_token}"}
    elif os.path.exists(ACCESS_TOK_PATH):
        with open(ACCESS_TOK_PATH) as f:
            access_refresh_dict = json.load(f)
        access = access_refresh_dict.get("access")
        refresh = access_refresh_dict.get("refresh")
        headers = {
            "Cookie": f"access={access};refresh={refresh}",
            "Content-Type": "application/json",
        }
    else:
        err = (
            "No https://biolm.ai credentials found. Please run "
            "`biolmai status` to debug."
        )
        raise AssertionError(err)
    return headers
