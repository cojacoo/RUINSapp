from typing import Callable, List
from functools import wraps
import hashlib


LOCAL = dict()

def _hashargs(fname, argnames):
    h = f'{fname};' + ','.join(argnames)
    digest = hashlib.sha256(h.encode()).hexdigest()

    return digest

def partial_memoize(hash_names: List[str], store: str = 'local'):
    def func_decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            argnames = [str(a) for a in args if a in hash_names]
            argnames.extend([str(v) for k, v in kwargs.items() if k in hash_names])
            
            # get the parameter hash
            h = _hashargs(f.__name__, argnames)
            
            # check if result exists
            if h in LOCAL:
                return LOCAL.get(h)
            else:
                # process
                result = f(*args, **kwargs)
                LOCAL[h] = result
                return result
        return wrapper
    return func_decorator
