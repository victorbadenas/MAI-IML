import numpy as np

def bytesToString(bytesObject):
    if hasattr(bytesObject, 'decode'):
        return bytesObject.decode()
    return bytesObject