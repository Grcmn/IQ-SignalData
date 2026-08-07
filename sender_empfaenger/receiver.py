
import numpy as np

def receive_array(s, a, dtype=np.complex64):
    # X_n(t) = a_n * s(t)
    return (a[:, None] * s[None, :]).astype(dtype)
