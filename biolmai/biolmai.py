"""Main module."""
import json
import os
import requests

import json, os, requests
import urllib3
import datetime
import time

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import logging

from biolmai.api import BASE_API_URL
from biolmai.const import ACCESS_TOK_PATH

log = logging.getLogger('biolm_util')


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=list(range(400, 599)),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def retry_minutes(sess, URL, HEADERS, dat, timeout, mins):
    """Retry for N minutes."""
    attempts, max_attempts = 0, 5
    try:
        now = datetime.datetime.now()
        try_until = now + datetime.timedelta(minutes=mins)
        while datetime.datetime.now() < try_until and attempts < max_attempts:
            response = None
            try:
                log.info('Trying {}'.format(datetime.datetime.now()))
                response = sess.post(
                    URL,
                    headers=HEADERS,
                    data=dat,
                    timeout=timeout
                )
                response.raise_for_status()
                if 'error' in response.json():
                    raise ValueError(response.json().dumps())
                else:
                    break
            except Exception as e:
                log.warning(e)
                if response:
                    log.warning(response.text)
                time.sleep(5)  # Wait 5 seconds between tries
            attempts += 1
        if response is None:
            err = "Got Nonetype response"
            raise ValueError(err)
        elif 'Server Error' in response.text:
            err = "Got Server Error"
            raise ValueError(err)
        else:
            response.raise_for_status()
    except Exception as e:
        raise
    else:
        return response


def get_user_auth_header():
    """Returns a dict with the appropriate Authorization header, either using
    an API token from BIOLMAI_TOKEN environment variable, or by reading the
    credentials file at ~/.biolmai/credntials next."""
    api_token = os.environ.get('BIOLMAI_TOKEN', None)
    if api_token:
        headers = {'Authorization': f'Token {api_token}'}
    elif os.path.exists(ACCESS_TOK_PATH):
        with open(ACCESS_TOK_PATH, 'r') as f:
            access_refresh_dict = json.load(f)
        access = access_refresh_dict.get('access')
        refresh = access_refresh_dict.get('refresh')
        headers = {
            'Cookie': 'access={};refresh={}'.format(access, refresh),
            'Content-Type': 'application/json'
        }
    else:
        AssertionError("No https://biolm.ai credentials found. Please run "
                       "`biolmai status` to debug.")
    return headers


def get_api_token():
    """Get a BioLM API token to use with future API requests.

    Copied from https://api.biolm.ai/#d7f87dfd-321f-45ae-99b6-eb203519ddeb.
    """
    url = "https://biolm.ai/api/auth/token/"

    payload = json.dumps({
        "username": os.environ.get("BIOLM_USER"),
        "password": os.environ.get("BIOLM_PASSWORD")
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()

    return response_json


def api_call(model_name, action, headers, payload):
    """Hit an arbitrary BioLM model inference API."""
    # Normally would POST multiple sequences at once for greater efficiency,
    # but for simplicity sake will do one at at time right now
    url = f'{BASE_API_URL}/models/{model_name}/{action}/'

    if not isinstance(payload, (list, dict)):
        err = "API request payload must be a list or dict, got {}"
        raise AssertionError(err.format(type(payload)))
    payload = json.dumps(payload)
    session = requests_retry_session()
    tout = urllib3.util.Timeout(total=180, read=180)
    response = retry_minutes(session, url, headers, payload, tout, mins=10)
    return response
