import numpy as np

SAMPLE_DTYPE = np.dtype("<i2")      # int16, little-endian, explizit
FULL_SCALE = 32767

# 1.5 * 2**14 = 24576, entspricht 2.5 dB Headroom gegenueber |s| = 1.

DEFAULT_SCALE = 1.5 * 2 ** 14

N_ANTENNAS_TOTAL = 8                # 8 Antennenkanäle, einer davon unbelegt
DEAD_ANTENNAS = (7,)                # Kanal ohne echtes Antennenelement


def scale_for(peak_amplitude, headroom_db=2.5):
    """skalierung mit headroom gegen clipping"""
    headroom = 10.0 ** (headroom_db / 20.0)
    return FULL_SCALE / (peak_amplitude * headroom)

class IQWriterInt16:

    def __init__(self, path, n_freq=1, n_antennas=N_ANTENNAS_TOTAL, dead_antennas=DEAD_ANTENNAS, scale=DEFAULT_SCALE):
        self.n_freq = int(n_freq)
        self.n_antennas = int(n_antennas)
        self.scale = float(scale)

        dead = sorted(set(int(i) for i in dead_antennas))

        self.dead_antennas = tuple(dead)
        self.live_antennas = tuple(i for i in range(self.n_antennas)
                                   if i not in self.dead_antennas)

        self.n_steps = 0 # geschriebe Zeitschritte
        self._fh = open(path, "wb")

    @property
    def n_live(self):
        return len(self.live_antennas)

    @property
    def bytes_per_step(self): # 32bytes bei 1 freq und 8 antennen
        return self.n_freq * self.n_antennas * 2 * SAMPLE_DTYPE.itemsize

    def write(self, x):
        
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[None, :, :] # none fügt achse der länge 1 ein
        
        # erzeugt ein array mit (n_freq, n_antennas, T) und nullt erstmal alle
        frame = np.zeros((self.n_freq, self.n_antennas, x.shape[2]), dtype=np.complex128)
        # nimmt indexe der live antennen und schreibt dort die werte von x rein, der "tote kanal" bleibt leer
        frame[:, self.live_antennas, :] = x

        # anordnung der achsen
        block = np.transpose(frame, (2, 0, 1))

        # Real/Imag als letzte Achse -> (T, n_freq, n_ant, 2)
        iq = np.stack([block.real, block.imag], axis=-1) * self.scale

        # rint: runden statt Richtung Null abschneiden
        # clip: Sättigung erzwingen, da Float->Int bei überlauf undefiniert ist
        out = np.clip(np.rint(iq), -FULL_SCALE - 1, FULL_SCALE).astype(SAMPLE_DTYPE)

        out.tofile(self._fh)
        self.n_steps += x.shape[2]

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
