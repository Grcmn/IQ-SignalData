"""
Ausbreitungskanal: zeitvariable fraktionale Verzögerung.

Kernidee des Generators: Ein Empfangspfad (Direktsignal, Zielecho, Clutter)
ist im komplexen Basisband exakt

    y(t) = A * s(t - tau(t)) * exp(-j*2*pi*f_c*tau(t))

mit dem Sendesignal s(t), der zeitvariablen Laufzeit tau(t) aus der
Geometrie und der Trägerfrequenz f_c. Der Dopplereffekt wird NICHT als
separater Frequenzoffset "aufmultipliziert", sondern entsteht automatisch
aus der Zeitabhängigkeit von tau(t):

    f_D(t) = -f_c * d(tau)/dt

Damit sind Doppler-Änderung über das CPI (Beschleunigung, Kurvenflug)
und Range-Migration (Ziel wandert während des CPI durch Range-Bins)
physikalisch korrekt enthalten.

Da tau(t)*fs praktisch nie ganzzahlig ist, wird s(t - tau) durch kubische
Lagrange-Interpolation (4 Stützstellen, Farrow-Struktur) ausgewertet.
"""

import numpy as np


def frac_delay(signal: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Wertet signal[idx] an nicht-ganzzahligen Indizes idx aus
    (kubische Lagrange-Interpolation). Indizes außerhalb des gültigen
    Bereichs liefern 0.
    """
    n = np.floor(idx).astype(np.int64)
    mu = idx - n

    valid = (n >= 1) & (n + 2 < len(signal))
    n_safe = np.clip(n, 1, len(signal) - 3)

    s_m1 = signal[n_safe - 1]
    s_0 = signal[n_safe]
    s_p1 = signal[n_safe + 1]
    s_p2 = signal[n_safe + 2]

    # Lagrange-Basispolynome für Stützstellen bei -1, 0, +1, +2
    c_m1 = -mu * (mu - 1) * (mu - 2) / 6
    c_0 = (mu + 1) * (mu - 1) * (mu - 2) / 2
    c_p1 = -(mu + 1) * mu * (mu - 2) / 2
    c_p2 = (mu + 1) * mu * (mu - 1) / 6

    y = c_m1 * s_m1 + c_0 * s_0 + c_p1 * s_p1 + c_p2 * s_p2
    y[~valid] = 0.0
    return y


def synth_path(tx_signal: np.ndarray, t: np.ndarray, fs: float, fc: float,
               tau: np.ndarray | float, power_w: float, preroll: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    Synthetisiert einen Empfangspfad.

    tx_signal: Sendesignal, beginnt bei t = -preroll/fs (Vorlauf, damit
               auch verzögerte Pfade am CPI-Anfang definiert sind)
    t:         Empfangszeitachse [s], Länge N
    tau:       Laufzeit [s], Skalar (statisch) oder Array der Länge N
    power_w:   mittlere Empfangsleistung des Pfads [W]
    preroll:   Vorlauf des Sendesignals in Samples

    Rückgabe: komplexes Basisbandsignal der Länge N.
    """
    tau = np.broadcast_to(np.asarray(tau, dtype=float), t.shape)

    # Sample-Index ins Sendesignal für den Zeitpunkt (t - tau)
    idx = (t - tau) * fs + preroll
    y = frac_delay(tx_signal, idx)

    # Trägerphasendrehung -> enthält Doppler und absolute Phasenlage
    y = y * np.exp(-2j * np.pi * fc * tau)

    # Amplitude aus Leistungsbilanz, zufällige Anfangsphase des Pfads
    phase0 = np.exp(2j * np.pi * rng.random())
    return np.sqrt(power_w) * phase0 * y
