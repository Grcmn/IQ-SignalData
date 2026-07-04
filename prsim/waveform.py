"""
Sendesignale der Illuminatoren of Opportunity als komplexe Basisbandsignale.

Drei Waveforms, entsprechend den drei Empfangsantennen des Systems:

  generate_dvbt() — DVB-T nach ETSI EN 300 744 (2k-Modus, 8-MHz-Kanal):
      1705 aktive Träger, 64-QAM-Nutzdaten, Scattered/Continual Pilots
      mit 4/3-Boost und PRBS-Modulation, TPS-Träger, Guard-Intervall.
      Pilotstruktur und Guard erzeugen deterministische Nebenmaxima in
      der Ambiguity-Funktion.

  generate_dab() — DAB nach ETSI EN 300 401 (Mode I, Band III):
      1536 Träger, pi/4-DQPSK, Guard-Intervall, Rahmenstruktur mit
      NULL-Symbol (96-ms-Rahmen). Das NULL-Symbol erzeugt periodische
      Artefakte in der Ambiguity-Funktion.

  generate_fm() — UKW-Rundfunk (Stereo-Multiplex):
      Audio (0-15 kHz) + 19-kHz-Pilotton + Stereo-Differenz um 38 kHz,
      Frequenzmodulation mit 75 kHz Hub. Die Radareigenschaften hängen
      vom Programminhalt ab (content="music" oder "speech") — der
      bekannteste Schwachpunkt FM-basierter Passivradare.

Warum kein weißes Rauschen als Ersatz? Die jeweilige Signalstruktur
bestimmt die Ambiguity-Funktion, mit der die Signalverarbeitung eines
echten Passivradars umgehen muss. Nur der Nutzinhalt (Bits bzw. Audio)
ist zufällig — für Radarzwecke irrelevant.
"""

import numpy as np

# Elementartakt eines 8-MHz-DVB-T-Kanals: 64/7 MHz
FS_DVBT = 64e6 / 7
# Elementartakt DAB (alle Modi): 2,048 MHz
FS_DAB = 2.048e6

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


# ═══════════════════════════════════════════════════════════════════════
# DAB (EN 300 401, Mode I)
# ═══════════════════════════════════════════════════════════════════════

_DAB_NFFT = 2048       # Tu = 2048 Samples bei 2,048 MHz (Mode I)
_DAB_K = 1536          # aktive Träger: -768..-1, +1..+768 (DC frei)
_DAB_NG = 504          # Guard-Intervall
_DAB_NULL = 2656       # Länge NULL-Symbol
_DAB_SYMS = 76         # OFDM-Symbole pro Rahmen (inkl. Phasenreferenz)
# Rahmenlänge: 2656 + 76*(2048+504) = 196608 Samples = 96 ms


def generate_dab(n_samples: int, seed: int | None = None) -> np.ndarray:
    """
    Erzeugt ein DAB-Basisbandsignal (Mode I) mit mindestens n_samples
    Samples bei fs = FS_DAB, normiert auf mittlere Leistung 1.

    Strukturell korrekt sind Rahmenaufbau (NULL-Symbol!), Trägerbelegung,
    Guard-Intervall und die differenzielle pi/4-DQPSK über die Symbole.
    Vereinfachung: das Phasenreferenzsymbol nutzt zufällige statt der
    genormten CAZAC-Phasen (für die Radar-Ambiguity unerheblich).
    """
    rng = np.random.default_rng(seed)
    sym_len = _DAB_NFFT + _DAB_NG
    frame_len = _DAB_NULL + _DAB_SYMS * sym_len
    n_frames = int(np.ceil(n_samples / frame_len))

    # Trägerindizes im FFT-Raster (DC bleibt frei)
    k = np.concatenate([np.arange(-_DAB_K // 2, 0), np.arange(1, _DAB_K // 2 + 1)])
    fft_bins = k % _DAB_NFFT

    out = np.zeros(n_frames * frame_len, dtype=np.complex128)
    pos = 0
    for _ in range(n_frames):
        pos += _DAB_NULL                       # NULL-Symbol: Sender aus
        # Phasenreferenzsymbol (Start der differenziellen Modulation)
        phase = np.exp(2j * np.pi * rng.integers(0, 4, _DAB_K) / 4)
        for _l in range(_DAB_SYMS):
            spec = np.zeros(_DAB_NFFT, dtype=np.complex128)
            spec[fft_bins] = phase
            sym = np.fft.ifft(spec) * _DAB_NFFT / np.sqrt(_DAB_K)
            out[pos:pos + _DAB_NG] = sym[-_DAB_NG:]
            out[pos + _DAB_NG:pos + sym_len] = sym
            pos += sym_len
            # pi/4-DQPSK: Phaseninkrement pi/4 + m*pi/2 pro Träger
            inc = np.pi / 4 + np.pi / 2 * rng.integers(0, 4, _DAB_K)
            phase = phase * np.exp(1j * inc)

    out = out[:n_samples]
    return out / np.sqrt(np.mean(np.abs(out) ** 2))


# ═══════════════════════════════════════════════════════════════════════
# UKW-Rundfunk (FM-Stereo-Multiplex)
# ═══════════════════════════════════════════════════════════════════════

def _lowpass_noise(n: int, cutoff_hz: float, fs: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Bandbegrenztes Gauß-Rauschen (Brickwall-Tiefpass im Frequenzbereich),
    normiert auf Standardabweichung 1."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[f > cutoff_hz] = 0.0
    x = np.fft.irfft(spec, n)
    return x / np.std(x)


def generate_fm(n_samples: int, fs: float, content: str = "music",
                deviation_hz: float = 75e3,
                seed: int | None = None) -> np.ndarray:
    """
    Erzeugt ein UKW-Rundfunksignal (Stereo-MPX, FM-moduliert) bei
    Abtastrate fs (empfohlen >= 300 kHz, Carson-Bandbreite ~256 kHz).
    Konstante Einhüllende, mittlere Leistung 1.

    content:
      "music"  — breitbandiges Audio (rauschartig): gutmütige,
                 schmale Ambiguity-Funktion
      "speech" — bandbegrenztes Audio mit Sprechpausen: zeitweise
                 fast unmodulierter Träger -> schlechte Range-Auflösung,
                 stark schwankende Radarleistung (Worst Case für FM-PCL)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    if content == "music":
        l_plus_r = _lowpass_noise(n_samples, 15e3, fs, rng)
        l_minus_r = _lowpass_noise(n_samples, 15e3, fs, rng)
    elif content == "speech":
        # Silbenrhythmus: Ein/Aus-Hüllkurve aus sehr langsamem Rauschen
        env = np.clip(_lowpass_noise(n_samples, 1.5, fs, rng), 0.0, None)
        l_plus_r = _lowpass_noise(n_samples, 4e3, fs, rng) * env
        rms = np.sqrt(np.mean(l_plus_r ** 2))
        l_plus_r /= max(rms, 1e-12)
        l_minus_r = np.zeros(n_samples)        # Sprache: mono
    else:
        raise ValueError(f"Unbekannter FM-Inhalt: {content!r}")

    # Stereo-Multiplex: Mono + 19-kHz-Pilot + DSB-Differenzsignal um 38 kHz
    mpx = (0.45 * l_plus_r
           + 0.09 * np.sin(2 * np.pi * 19e3 * t)
           + 0.45 * l_minus_r * np.sin(2 * np.pi * 38e3 * t))
    mpx /= np.max(np.abs(mpx)) + 1e-12         # Spitzenhub = deviation_hz

    phase = 2 * np.pi * deviation_hz * np.cumsum(mpx) / fs
    return np.exp(1j * phase)
