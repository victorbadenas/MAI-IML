import numpy as np
import pandas as pd
import time

def bytesToString(bytesObject):
    if hasattr(bytesObject, 'decode'):
        return bytesObject.decode()
    return bytesObject

def timer(print_=False):
    def inner2(func):
        def inner(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            if print_:
                print(f"{func.__name__} ran in {time.time()-st:.2f}s")
                return ret
            else:
                delta = time.time() - st
                return ret, delta
        return inner
    return inner2

def convertToNumpy(data):
    """
    Converts data to numpy if the data is an object of pd.DataFrame or a list
    of lists. Otherwise, raise ValueError.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
        if len(data) == 2:
            return data
        else:
            raise ValueError(f"Expected a 2D list as input")
    raise ValueError(f"type {type(data)} not supported")