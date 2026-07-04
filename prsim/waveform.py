"""
DVB-T-Sendesignal (Illuminator of Opportunity) als komplexes Basisbandsignal.

Implementiert einen vereinfachten, aber strukturell korrekten DVB-T-Modulator
nach ETSI EN 300 744 (2k-Modus, 8-MHz-Kanal):

- FFT-Länge 2048, davon 1705 aktive Träger
- 64-QAM-Nutzdaten (zufällige Bits — der Inhalt ist für Radar irrelevant,
  die *Struktur* nicht)
- Scattered Pilots (alle 12 Träger, Offset rotiert pro Symbol) und
  Continual Pilots mit 4/3-Boost, BPSK-moduliert mit der PRBS aus der Norm
- TPS-Träger (hier: zufälliges BPSK, Inhalt irrelevant)
- Guard-Intervall als zyklisches Präfix

Warum kein weißes Rauschen als Ersatz? Die Pilotstruktur und das
Guard-Intervall erzeugen deterministische Nebenmaxima in der
Ambiguity-Funktion, mit denen die Signalverarbeitung eines echten
Passivradars umgehen muss. Ein Generator, der das nicht abbildet,
testet die Algorithmen nicht realistisch.
"""

import numpy as np

# Elementartakt eines 8-MHz-DVB-T-Kanals: 64/7 MHz
FS_DVBT = 64e6 / 7

_NFFT = 2048          # 2k-Modus
_K = 1705             # aktive Träger (Index 0..1704, Mitte = 852)
_K_CENTER = 852

# Continual-Pilot-Positionen im 2k-Modus (EN 300 744, Tab. 9)
_CONTINUAL_PILOTS = np.array([
    0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450,
    483, 525, 531, 618, 636, 714, 759, 765, 780, 804, 873, 888, 918,
    939, 942, 969, 984, 1050, 1101, 1107, 1110, 1137, 1140, 1146,
    1206, 1269, 1323, 1377, 1491, 1683, 1704])

# TPS-Träger im 2k-Modus (EN 300 744, Tab. 8)
_TPS_CARRIERS = np.array([
    34, 50, 209, 346, 413, 569, 595, 688, 790, 901,
    1073, 1219, 1262, 1286, 1469, 1594, 1687])


def _pilot_prbs(n: int) -> np.ndarray:
    """PRBS-Referenzsequenz w_k, Generatorpolynom X^11 + X^2 + 1,
    Initialisierung mit Einsen (EN 300 744, Kap. 4.5.2)."""
    reg = np.ones(11, dtype=np.int64)
    w = np.empty(n, dtype=np.int64)
    for i in range(n):
        w[i] = reg[10]
        fb = reg[10] ^ reg[1]
        reg[1:] = reg[:-1]
        reg[0] = fb
    return w


def generate_dvbt(n_samples: int, guard: float = 1 / 8,
                  seed: int | None = None) -> np.ndarray:
    """
    Erzeugt ein DVB-T-Basisbandsignal mit mindestens n_samples Samples
    bei fs = FS_DVBT, normiert auf mittlere Leistung 1.

    guard: Guard-Intervall-Anteil (1/4, 1/8, 1/16 oder 1/32)
    """
    rng = np.random.default_rng(seed)
    ng = int(_NFFT * guard)
    sym_len = _NFFT + ng
    n_syms = int(np.ceil(n_samples / sym_len))

    # Pilotwerte: BPSK aus PRBS, Leistungs-Boost 16/9 (Amplitude 4/3)
    w = _pilot_prbs(_K)
    pilot_val = 4 / 3 * (1.0 - 2.0 * w)

    carrier_idx = np.arange(_K)
    fft_bins = (carrier_idx - _K_CENTER) % _NFFT  # Träger symmetrisch um DC

    out = np.empty(n_syms * sym_len, dtype=np.complex128)
    qam_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42.0)

    for l in range(n_syms):
        # Scattered Pilots: k mod 12 == 3*(l mod 4)
        scattered = np.arange(3 * (l % 4), _K, 12)
        pilots = np.union1d(scattered, _CONTINUAL_PILOTS)
        data = np.setdiff1d(carrier_idx, np.union1d(pilots, _TPS_CARRIERS))

        carriers = np.zeros(_K, dtype=np.complex128)
        carriers[data] = (rng.choice(qam_levels, size=data.size)
                          + 1j * rng.choice(qam_levels, size=data.size))
        carriers[pilots] = pilot_val[pilots]
        carriers[_TPS_CARRIERS] = 1.0 - 2.0 * rng.integers(0, 2, _TPS_CARRIERS.size)

        spec = np.zeros(_NFFT, dtype=np.complex128)
        spec[fft_bins] = carriers
        sym = np.fft.ifft(spec) * _NFFT / np.sqrt(_K)

        out[l * sym_len: l * sym_len + ng] = sym[-ng:]      # zyklisches Präfix
        out[l * sym_len + ng: (l + 1) * sym_len] = sym

    out = out[:n_samples]
    return out / np.sqrt(np.mean(np.abs(out) ** 2))
