from __future__ import absolute_import

import inspect

try: from functools import lru_cache  # Python 3
except ImportError:
    def lru_cache(*args):  # Python 2: no-op wrapper
        return lambda f: f

try: getargspec = inspect.getfullargspec  # Python 3
except AttributeError:
    getargspec = inspect.getargspec  # Python 2

@lru_cache(100)
def get_function_defaults(for_func):
    """ Get a dict of function's arguments that have default values """
    # Analyze the method
    argspec = getargspec(for_func)  # TODO: use signature() in Python 3.3

    # Get the names of the kwargs
    # Only process those that have defaults
    n_args = len(argspec.args) - len(argspec.defaults or ())  # Args without defaults
    kwargs_names = argspec.args[n_args:]

    # Get defaults for kwargs: put together argument names + default values
    defaults = dict(zip(kwargs_names, argspec.defaults or ()))

    # Done
    return defaults


def pluck_kwargs_from(dct, for_func):
    """ Analyze a function, pluck the arguments it needs from a dict """
    defaults = get_function_defaults(for_func)

    # Get the values for these kwargs
    kwargs = {k: dct.get(k, defaults[k])
              for k in defaults.keys()}

    # Done
    return kwargs
