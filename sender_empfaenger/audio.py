
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def load_audio(path, fs, duration_s=None):

    import soundfile as sf

    data, fs_file = sf.read(path, dtype="float64", always_2d=True)

    left = data[:, 0]
    right = data[:, 1] if data.shape[1] >= 2 else data[:, 0]

    left = _resample(left, fs_file, fs)
    right = _resample(right, fs_file, fs)

    if duration_s is not None:
        m = int(round(duration_s * fs))
        left, right = left[:m], right[:m]

    return _normalize(left, right)


def _resample(x, fs_in, fs_out):

    if int(fs_in) == int(fs_out):
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(int(fs_out), int(fs_in))
    return resample_poly(x, ratio.numerator, ratio.denominator)


def _normalize(left, right):

    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak == 0.0:
        return left, right
    return left / peak, right / peak