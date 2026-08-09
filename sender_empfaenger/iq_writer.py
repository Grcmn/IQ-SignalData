
import json
from pathlib import Path

import numpy as np

AXIS_ORDER = ["time", "iq", "antenna"]
SAMPLE_DTYPE = np.float32


class IQWriter:
    """Schreibt Bloecke der Form (n_antennas, T) fortlaufend in eine .dat.

    Layout auf der Platte: [time][iq][antenna] als float32 in C-Order, d.h.
    pro Zeitschritt erst alle I-Werte (Antenne 0..n-1), dann alle Q-Werte.
    Die Datei selbst traegt keine Form -- die steht im JSON-Sidecar daneben.
    """

    def __init__(self, path, n_antennas, meta=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_antennas = int(n_antennas)
        self.meta = dict(meta or {})

        self._fh = open(self.path, "wb")
        self._n = 0          # bisher geschriebene Zeitschritte

    @property
    def sidecar_path(self):
        return self.path.with_suffix(".json")

    @property
    def n_written(self):
        return self._n

    def write(self, x):
        """x: (n_antennas, T) komplex. Gibt die Anzahl geschriebener Samples zurueck."""
        if self._fh is None:
            raise RuntimeError("Writer ist bereits geschlossen")
        if x.ndim != 2 or x.shape[0] != self.n_antennas:
            raise ValueError(f"erwarte ({self.n_antennas}, T), bekommen {x.shape}")

        # (n_ant, T) -> (T, n_ant), zusammenhaengend, damit .view() erlaubt ist
        block = np.ascontiguousarray(x.T, dtype=np.complex64)
        count = block.shape[0]

        # complex64 -> (T, n_ant, 2) mit I,Q als letzter Achse
        iq = block.view(SAMPLE_DTYPE).reshape(count, self.n_antennas, 2)

        # -> (T, 2, n_ant). ascontiguousarray ist Pflicht: nach transpose ist das
        # nur ein View mit vertauschten Strides und tofile() wuerde die alte
        # Reihenfolge schreiben, ohne zu meckern.
        out = np.ascontiguousarray(iq.transpose(0, 2, 1))
        out.tofile(self._fh)

        self._n += count
        return count

    def close(self):
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None

        meta = {
            "filename": self.path.name,
            "dtype": "float32",
            "axis_order": AXIS_ORDER,
            "shape": [self._n, 2, self.n_antennas],
            "n_samples": self._n,
            "n_antennas": self.n_antennas,
        }
        meta.update(self.meta)
        if "fs_hz" in meta and meta["fs_hz"]:
            meta["duration_s"] = self._n / float(meta["fs_hz"])

        self.sidecar_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_iq(path):
    """Liest eine mit IQWriter erzeugte .dat zurueck.

    Gibt (x, meta) zurueck, x hat die Form (T, n_antennas) und ist komplex --
    also transponiert gegenueber dem, was receive_array liefert.
    """
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    n_ant = int(meta["n_antennas"])

    raw = np.fromfile(path, dtype=SAMPLE_DTYPE)
    if raw.size % (2 * n_ant) != 0:
        raise ValueError("Dateilaenge passt nicht zu n_antennas aus dem Sidecar")

    a = raw.reshape(-1, 2, n_ant)
    return (a[:, 0, :] + 1j * a[:, 1, :]).astype(np.complex64), meta
