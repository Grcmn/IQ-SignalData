
import numpy as np

PILOT_HZ = 19e3
DEVIATION_HZ = 75e3


class FMStream:
    
    def __init__(self, fs, left, right, deviation_hz=DEVIATION_HZ,
                 pilot_hz=PILOT_HZ):
        if len(left) != len(right):
            raise ValueError("l und r ungleich")

        self.fs = float(fs)
        self.left = np.asarray(left, dtype=np.float64)
        self.right = np.asarray(right, dtype=np.float64)
        self.deviation_hz = float(deviation_hz)
        self.pilot_hz = float(pilot_hz)

        self._n = 0          # laufender Sample-Index
        self._phase = 0.0    # Restphase am Ende des letzten Blocks [rad]

    @property
    def n_samples(self):
        """Verfuegbare Samples insgesamt (= Laenge des Audiomaterials)."""
        return self.left.size

    @property
    def remaining(self):
        """Noch nicht ausgegebene Samples."""
        return self.n_samples - self._n

    def _multiplex(self, n0, count):
        t = (n0 + np.arange(count)) / self.fs
        l = self.left[n0:n0 + count]
        r = self.right[n0:n0 + count]

        mono = 0.5 * (l + r)                               # 0 .. 15 kHz
        diff = 0.5 * (l - r)                               # -> DSB um 38 kHz
        subcarrier = np.sin(4.0 * np.pi * self.pilot_hz * t)
        pilot = np.sin(2.0 * np.pi * self.pilot_hz * t)

        return 0.9 * (mono + diff * subcarrier) + 0.1 * pilot

    def next_block(self, block_size):
    
        if self.remaining < block_size:
            return np.empty(0, dtype=np.complex128)

        m = self._multiplex(self._n, block_size)
        self._n += block_size

        inst_freq = self.deviation_hz * m
        phase = self._phase + 2.0 * np.pi * np.cumsum(inst_freq) / self.fs
        self._phase = float(phase[-1])

        return np.exp(1j * phase)