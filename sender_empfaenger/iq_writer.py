"""IQ-Persistenz im SiL-Format.

Layout: C-Array (row-major) der Form [time][freq][antenna] mit comp_int16,
d. h. je Element zwei int16 (Realteil, dann Imaginaerteil), little-endian.

Der letzte Index laeuft am schnellsten. Ein Zeitschritt belegt
n_freq * n_ant * 4 Byte; die Adresse eines Elements ist damit

    byte_offset(t, f, a) = ((t * n_freq + f) * n_ant + a) * 4

Die Recordgroesse ist fest, auch wenn nur ein Frequenzslot belegt ist --
nicht belegte Slots werden mit Null gefuellt. Nur so bleibt die
Adressformel auf der C-Seite gueltig.
"""

import json
from pathlib import Path

import numpy as np

AXIS_ORDER = ["time", "freq", "antenna"]
SAMPLE_DTYPE = np.dtype("<i2")      # int16, little-endian, explizit
N_FREQ_SLOTS = 16
FULL_SCALE = 32767
FORMAT_VERSION = 2

# 2**14: 6 dB Headroom gegenueber |s| = 1, damit die spaetere Summe aus
# Direktpfad + Echo + Rauschen nicht clippt. Fest ueber die ganze Aufnahme --
# eine blockweise Normierung wuerde die Amplitudeninformation zerstoeren.
DEFAULT_SCALE = 2 ** 14


class IQWriterInt16:
    """Schreibt Bloecke als comp_int16 im Layout [time][freq][antenna]."""

    def __init__(self, path, n_antennas, n_freq=N_FREQ_SLOTS,
                 scale=DEFAULT_SCALE, meta=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_antennas = int(n_antennas)
        self.n_freq = int(n_freq)
        self.scale = float(scale)
        self.meta = dict(meta or {})

        self._fh = open(self.path, "wb")
        self._n = 0             # geschriebene Zeitschritte
        self._n_clipped = 0     # geclippte int16-Werte (soll 0 bleiben)
        self._peak = 0.0        # groesster Betrag vor Skalierung

    @property
    def sidecar_path(self):
        return self.path.with_suffix(".json")

    @property
    def n_written(self):
        return self._n

    @property
    def bytes_per_sample(self):
        return self.n_freq * self.n_antennas * 4

    def write(self, x, freq_index=0):
        """Ein Block auf einem einzelnen Frequenzslot.

        x : (n_ant, T) complex   -- uebrige Slots werden genullt.
        """
        if x.ndim != 2 or x.shape[0] != self.n_antennas:
            raise ValueError(f"erwarte ({self.n_antennas}, T), bekommen {x.shape}")
        if not 0 <= freq_index < self.n_freq:
            raise ValueError(f"freq_index {freq_index} ausserhalb 0..{self.n_freq-1}")

        count = x.shape[1]
        full = np.zeros((self.n_freq, self.n_antennas, count), dtype=np.complex128)
        full[freq_index] = x
        return self.write_multi(full)

    def write_multi(self, x):
        """Ein Block ueber alle Frequenzslots.

        x : (n_freq, n_ant, T) complex
        """
        if self._fh is None:
            raise RuntimeError("Writer ist bereits geschlossen")
        expected = (self.n_freq, self.n_antennas)
        if x.ndim != 3 or x.shape[:2] != expected:
            raise ValueError(f"erwarte {expected + ('T',)}, bekommen {x.shape}")

        count = x.shape[2]
        self._peak = max(self._peak, float(np.abs(x).max(initial=0.0)))

        # (n_freq, n_ant, T) -> (T, n_freq, n_ant), damit die Antenne am
        # schnellsten laeuft (C-Order, letzter Index innen).
        block = np.transpose(x, (2, 0, 1))

        # Real/Imag als letzte Achse -> (T, n_freq, n_ant, 2).
        # Genau das Speicherbild eines struct {int16 re, im;}.
        iq = np.stack([block.real, block.imag], axis=-1) * self.scale

        # rint statt astype: astype schneidet Richtung Null ab und wrappt bei
        # Ueberlauf still um. Erst runden, dann saettigen, und das Clipping zaehlen.
        rounded = np.rint(iq)
        self._n_clipped += int(np.count_nonzero(
            (rounded > FULL_SCALE) | (rounded < -FULL_SCALE - 1)))
        out = np.ascontiguousarray(
            np.clip(rounded, -FULL_SCALE - 1, FULL_SCALE).astype(SAMPLE_DTYPE))

        out.tofile(self._fh)
        self._n += count
        return count

    def close(self):
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None

        meta = {
            "format_version": FORMAT_VERSION,
            "filename": self.path.name,
            "dtype": "comp_int16",
            "component_dtype": "int16",
            "byte_order": "little",
            "array_order": "C",
            "axis_order": AXIS_ORDER,
            "shape": [self._n, self.n_freq, self.n_antennas],
            "n_samples": self._n,
            "n_freq_slots": self.n_freq,
            "n_antennas": self.n_antennas,
            "bytes_per_element": 4,
            "bytes_per_time_step": self.bytes_per_sample,
            "scale": self.scale,
            "full_scale": FULL_SCALE,
            "n_clipped": self._n_clipped,
            "peak_before_scaling": self._peak,
        }
        meta.update(self.meta)
        if meta.get("fs_hz"):
            meta["duration_s"] = self._n / float(meta["fs_hz"])

        self.sidecar_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_iq_int16(path, freq_index=None, as_float=True):
    """Liest die Datei zurueck.

    freq_index=None -> (T, n_freq, n_ant);  sonst (T, n_ant) fuer diesen Slot.
    as_float=False  -> rohe int16 mit zusaetzlicher Achse 2 (re, im).
    """
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    n_ant = int(meta["n_antennas"])
    n_freq = int(meta["n_freq_slots"])
    scale = float(meta["scale"])

    raw = np.fromfile(path, dtype=SAMPLE_DTYPE)
    per_step = n_freq * n_ant * 2
    if raw.size % per_step != 0:
        raise ValueError("Dateilaenge passt nicht zu n_freq_slots/n_antennas")

    a = raw.reshape(-1, n_freq, n_ant, 2)
    if freq_index is not None:
        a = a[:, freq_index]
    if not as_float:
        return a, meta
    return ((a[..., 0] + 1j * a[..., 1]) / scale).astype(np.complex64), meta


def byte_offset(t, f, a, n_freq, n_ant):
    """Adressformel, identisch zu der auf der C-Seite."""
    return ((t * n_freq + f) * n_ant + a) * 4