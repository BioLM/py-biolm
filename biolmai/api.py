"""References to API endpoints."""
from __future__ import annotations

from biolmai import biolmai
import inspect
from biolmai.const import BASE_DOMAIN, MULTIPROCESS_THREADS
from functools import lru_cache
from biolmai.validate import UnambiguousAA

BASE_API_URL = f'{BASE_DOMAIN}/api/v1'

def validate_endpoint_payload(f):
    # Extract sequences or other data to be used in JSON POST payload
    def wrapper(*args, **kwargs):
        class_obj_self = args[0]
        input_data = args[1]
        if not isinstance(input_data, str):
            err = "Input sequence must be a DNA or protein string"
            raise ValueError(err)

        for c in class_obj_self.seq_classes:
            c()(input_data)  # Validate input data against regex
        return f(*args, **kwargs)
    return wrapper


@lru_cache(maxsize=64)
def validate_endpoint_action(f):
    """Decorator for APIEndpoint subclass methods, like `predict()`,
    `generate()`, etc. to make sure each endpoint allows the action called by
     the user."""
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
            if action_method_name not in class_obj_self.action_class_strings:
                err = 'Only {} supported on {}'
                err = err.format(
                    class_obj_self.action_class_strings,
                    class_obj_self.__class__.__name__
                )
                raise AssertionError(err)


        return f(*args, **kwargs)
    return wrapper


class APIEndpoint(object):
    def __init__(self, multiprocess_threads=None):
        # Check for instance-specific threads, otherwise read from env var
        if multiprocess_threads is not None:
            self.multiprocess_threads = multiprocess_threads
        else:
            self.multiprocess_threads = MULTIPROCESS_THREADS  # Could be False
        # Get correct auth-like headers
        self.auth_headers = biolmai.get_user_auth_header()
        self.action_class_strings = [
            c.__name__.replace('Action', '').lower() for c in self.action_classes
        ]


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @validate_endpoint_action
    @validate_endpoint_payload
    def predict(self, dat):
        payload = {"instances": [{"data": {"text": dat}}]}
        resp = biolmai.api_call(
            model_name=self.slug,
            headers=self.auth_headers,  # From APIEndpoint base class
            action='predict',
            payload=payload
        )
        return resp.json()

    def infer(self, dat):
        return self.predict(dat)
