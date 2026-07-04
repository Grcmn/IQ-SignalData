"""
Szenariogeometrie und Leistungsbilanz.

Alle Positionen in kartesischen Koordinaten [x, y, z] in Metern
(lokales ENU-System, z = Höhe). Zeiten in Sekunden, Leistungen in Watt.
"""

from dataclasses import dataclass, field
import numpy as np

C0 = 299792458.0        # Lichtgeschwindigkeit [m/s]
K_BOLTZ = 1.380649e-23  # Boltzmann-Konstante [J/K]
T0 = 290.0              # Referenztemperatur [K]


@dataclass
class Transmitter:
    """Illuminator of Opportunity, z.B. DVB-T-Sendemast."""
    pos: np.ndarray                 # [m]
    erp_w: float = 50e3             # effektive Strahlungsleistung [W]
    freq_hz: float = 626e6          # Trägerfrequenz (z.B. Kanal 40) [Hz]

    @property
    def wavelength(self) -> float:
        return C0 / self.freq_hz


@dataclass
class Receiver:
    """Passivradar-Empfänger mit Referenz- und Überwachungsantenne."""
    pos: np.ndarray
    gain_surv_db: float = 8.0       # Gewinn Überwachungsantenne [dBi]
    gain_ref_db: float = 12.0       # Gewinn Referenzantenne (auf TX gerichtet) [dBi]
    noise_figure_db: float = 6.0    # Rauschzahl [dB]
    dpi_suppression_db: float = 40.0  # Unterdrückung des Direktsignals im
                                      # Überwachungskanal (Antennen-Null /
                                      # Beamforming Richtung Sender) [dB]


@dataclass
class Target:
    """Bewegtes Punktziel mit geradliniger Trajektorie p(t) = pos0 + vel*t."""
    name: str
    pos0: np.ndarray                # Position bei t=0 [m]
    vel: np.ndarray                 # Geschwindigkeitsvektor [m/s]
    rcs_m2: float = 20.0            # bistatischer Radarquerschnitt [m^2]

    def pos(self, t):
        """Position(en) zu Zeitpunkt(en) t; t darf Skalar oder Array sein."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.pos0[None, :] + t[:, None] * self.vel[None, :]


@dataclass
class StaticScatterer:
    """Stationärer Reflektor (Gebäude, Windrad, Gelände) -> Clutter bei 0 Hz."""
    name: str
    pos: np.ndarray
    rcs_m2: float = 1000.0


# ── Geometrische Größen ────────────────────────────────────────────────

def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.atleast_2d(a) - np.atleast_2d(b), axis=-1)


def bistatic_delay(tx: Transmitter, rx: Receiver, positions: np.ndarray) -> np.ndarray:
    """Absolute Laufzeit Sender -> Ziel -> Empfänger [s]."""
    r_tx = _dist(positions, tx.pos)
    r_rx = _dist(positions, rx.pos)
    return (r_tx + r_rx) / C0


def baseline(tx: Transmitter, rx: Receiver) -> float:
    """Länge der Basislinie Sender-Empfänger [m]."""
    return float(np.linalg.norm(tx.pos - rx.pos))


def bistatic_range(tx: Transmitter, rx: Receiver, positions: np.ndarray) -> np.ndarray:
    """Bistatische Range R_tx + R_rx - L [m] — das, was die CAF anzeigt,
    wenn mit dem Referenzkanal (Direktsignal) korreliert wird."""
    r_tx = _dist(positions, tx.pos)
    r_rx = _dist(positions, rx.pos)
    return r_tx + r_rx - baseline(tx, rx)


def doppler_hz(tx: Transmitter, rx: Receiver, target: Target, t: float) -> float:
    """Bistatische Dopplerfrequenz f_D = -(1/lambda) d(R_tx+R_rx)/dt zum
    Zeitpunkt t (numerische Ableitung)."""
    dt = 1e-3
    tau1 = bistatic_delay(tx, rx, target.pos(t - dt))[0]
    tau2 = bistatic_delay(tx, rx, target.pos(t + dt))[0]
    return -tx.freq_hz * (tau2 - tau1) / (2 * dt)


# ── Leistungsbilanz ────────────────────────────────────────────────────

def direct_rx_power(tx: Transmitter, rx: Receiver, gain_rx_db: float) -> float:
    """Empfangsleistung des Direktsignals (Friis) [W]."""
    lam = tx.wavelength
    L = baseline(tx, rx)
    g_rx = 10 ** (gain_rx_db / 10)
    return tx.erp_w * g_rx * lam ** 2 / ((4 * np.pi) ** 2 * L ** 2)


def target_rx_power(tx: Transmitter, rx: Receiver, pos: np.ndarray,
                    rcs_m2: float, gain_rx_db: float) -> float:
    """Empfangsleistung eines Zielechos, bistatische Radargleichung [W]."""
    lam = tx.wavelength
    r_tx = _dist(pos, tx.pos)[0]
    r_rx = _dist(pos, rx.pos)[0]
    g_rx = 10 ** (gain_rx_db / 10)
    return (tx.erp_w * g_rx * lam ** 2 * rcs_m2
            / ((4 * np.pi) ** 3 * r_tx ** 2 * r_rx ** 2))


def noise_power(fs: float, noise_figure_db: float) -> float:
    """Thermische Rauschleistung im komplexen Basisband der Bandbreite fs:
    N = k*T0*B*F [W]."""
    return K_BOLTZ * T0 * fs * 10 ** (noise_figure_db / 10)
