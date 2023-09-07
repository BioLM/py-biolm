"""References to API endpoints."""
from __future__ import annotations

from biolmai import biolmai
import inspect
from biolmai.const import BASE_DOMAIN, MULTIPROCESS_THREADS
from functools import lru_cache
from biolmai.validate import UnambiguousAA

BASE_API_URL = f'{BASE_DOMAIN}/api/v1'

@lru_cache(maxsize=64)
def validate_endpoint_action(allowed_classes, method_name, api_class_name):
    action_method_name = method_name.split('.')[-1]
    if action_method_name not in allowed_classes:
        err = 'Only {} supported on {}'
        err = err.format(
            list(allowed_classes),
            api_class_name
        )
        raise AssertionError(err)

def validate(f):
    def wrapper(*args, **kwargs):
        # Get class instance at runtime, so you can access not just
        # APIEndpoints, but any *parent* classes of that,
        # like ESMFoldSinglechain.
        class_obj_self = args[0]
        try:
            is_method = inspect.getfullargspec(f)[0][0] == 'self'
        except:
            is_method = False

        if is_method:
            name = '{}.{}.{}'.format(f.__module__, args[0].__class__.__name__,
                                     f.__name__)
        else:
            name = '{}.{}'.format(f.__module__, f.__name__)

        if is_method:
            # Splits e.g. 'biolmai.api.ESMFoldSingleChain.predict'
            action_method_name = name.split('.')[-1]
            validate_endpoint_action(
                class_obj_self.action_class_strings,
                action_method_name,
                class_obj_self.__class__.__name__
            )

        input_data = args[1]
        if not isinstance(input_data, str):
            err = "Input sequence must be a DNA or protein string"
            raise ValueError(err)

        for c in class_obj_self.seq_classes:
            c()(input_data)  # Validate input data against regex
        res = f(*args, **kwargs)
        return res.json()
    return wrapper


def validate_endpoint_payload(f):
    # Extract sequences or other data to be used in JSON POST payload
    def wrapper(*args, **kwargs):
        class_obj_self = args[0]
        return f(*args, **kwargs)
    return wrapper


class APIEndpoint(object):
    batch_size = 3  # Overwrite in parent classes as needed

    def __init__(self, multiprocess_threads=None):
        # Check for instance-specific threads, otherwise read from env var
        if multiprocess_threads is not None:
            self.multiprocess_threads = multiprocess_threads
        else:
            self.multiprocess_threads = MULTIPROCESS_THREADS  # Could be False
        # Get correct auth-like headers
        self.auth_headers = biolmai.get_user_auth_header()
        self.action_class_strings = tuple([
            c.__name__.replace('Action', '').lower() for c in self.action_classes
        ])

    @validate
    def predict(self, dat):
        payload = {"instances": [{"data": {"text": dat}}]}
        resp = biolmai.api_call(
            model_name=self.slug,
            headers=self.auth_headers,  # From APIEndpoint base class
            action='predict',
            payload=payload
        )
        return resp

    def infer(self, dat):
        return self.predict(dat)

    @validate
    def tokenize(self, dat):
        payload = {"instances": [{"data": {"text": dat}}]}
        resp = biolmai.api_call(
            model_name=self.slug,
            headers=self.auth_headers,  # From APIEndpoint base class
            action='tokenize',
            payload=payload
        )
        return resp


class PredictAction(object):

    def __str__(self):
        return 'PredictAction'

class GenerateAction(object):

    def __str__(self):
        return 'GenerateAction'

class TokenizeAction(object):

    def __str__(self):
        return 'TokenizeAction'

class ExplainAction(object):

    def __str__(self):
        return 'ExplainAction'

class SimilarityAction(object):

    def __str__(self):
        return 'SimilarityAction'


class FinetuneAction(object):

    def __str__(self):
        return 'FinetuneAction'


class ESMFoldSingleChain(APIEndpoint):
    slug = 'esmfold-singlechain'
    action_classes = (PredictAction, )
    seq_classes = (UnambiguousAA, )
    batch_size = 2
