"""
Empfänger-Impairments: Rauschen, LO-Phasenrauschen, ADC-Quantisierung.
"""

import numpy as np


def awgn(n: int, power_w: float, rng: np.random.Generator) -> np.ndarray:
    """Komplexes weißes Gauß-Rauschen mit mittlerer Leistung power_w."""
    return np.sqrt(power_w / 2) * (rng.standard_normal(n)
                                   + 1j * rng.standard_normal(n))


def lo_phase_noise(n: int, fs: float, linewidth_hz: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Multiplikativer Phasenrausch-Term exp(j*phi(t)) eines Oszillators,
    modelliert als Wiener-Prozess (Lorentz-Linienform mit 3-dB-Linienbreite
    linewidth_hz). Bei gemeinsamem LO für Referenz- und Überwachungskanal
    denselben Term auf beide Kanäle anwenden — dann kürzt er sich in der
    Kreuzkorrelation weitgehend heraus (genau wie in echter Hardware).
    """
    var_step = 2 * np.pi * linewidth_hz / fs
    phi = np.cumsum(np.sqrt(var_step) * rng.standard_normal(n))
    return np.exp(1j * phi)


def adc_quantize(x: np.ndarray, bits: int = 12,
                 headroom_db: float = 12.0) -> tuple[np.ndarray, float]:
    """
    Simuliert den ADC: Skalierung auf den Aussteuerbereich, Clipping,
    Quantisierung von I und Q auf 'bits' Bit.

    headroom_db: Abstand zwischen RMS des Signals und Vollaussteuerung
                 (Crest-Faktor-Reserve; OFDM braucht >= 10-12 dB).

    Rückgabe: (quantisierte Integer-Codes als complex128, Skalierungsfaktor
    'scale' in Volt/LSB-Analogie). x ≈ codes * scale.
    """
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    fullscale = rms * 10 ** (headroom_db / 20)
    n_levels = 2 ** (bits - 1)
    scale = fullscale / n_levels

    def q(v):
        codes = np.round(v / scale)
        return np.clip(codes, -n_levels, n_levels - 1)

    return q(x.real) + 1j * q(x.imag), scale
