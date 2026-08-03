"""
Sender (Transmitter) — Schritt 1: Erzeugung des FM-Sendesignals.

Modelliert einen UKW-Rundfunksender als Illuminator of Opportunity.
Ausgegeben wird das komplexe Basisbandsignal (I/Q), so wie es die
Sendeantenne verlaesst — noch ohne Ausbreitung, Rauschen oder Empfaenger.

Signalkette im Sender:
    Audio (L, R)  ->  Stereo-Multiplex (MPX)  ->  FM-Modulation  ->  I/Q

Der Stereo-Multiplex (MPX) besteht aus:
    - Mono-Summensignal L+R           (0 .. 15 kHz)
    - 19-kHz-Pilotton                 (Referenz fuer den Stereo-Decoder)
    - Stereo-Differenz L-R als DSB    (um 38 kHz, = 2 x Pilot)

Das MPX-Signal wird anschliessend frequenzmoduliert (75 kHz Hub).
Ergebnis: ein Signal mit konstanter Einhuellender (|s(t)| = 1) und
mittlerer Leistung 1.
"""

import numpy as np

# Kennwerte des UKW-Rundfunks
PILOT_HZ = 19e3            # Stereo-Pilotton
SUBCARRIER_HZ = 38e3       # Traeger des Stereo-Differenzsignals (2 x Pilot)
AUDIO_CUTOFF_HZ = 15e3     # obere Audiogrenzfrequenz
DEVIATION_HZ = 75e3        # maximaler Frequenzhub


def _lowpass_noise(n, cutoff_hz, fs, rng):
    """Bandbegrenztes Gauss-Rauschen als Ersatz fuer echtes Audio.

    Fuer Radarzwecke ist nur die Signal*struktur* relevant, nicht der
    konkrete Audioinhalt — breitbandiges Rauschen bis cutoff_hz ist ein
    gutes Modell fuer typisches Musikprogramm. Normiert auf Standard-
    abweichung 1.
    """
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[f > cutoff_hz] = 0.0
    x = np.fft.irfft(spec, n)
    return x / np.std(x)


def generate_fm(n_samples, fs, seed=None, return_mpx=False):
    """Erzeugt ein FM-Rundfunk-Sendesignal im komplexen Basisband.

    Parameter
    ---------
    n_samples : int
        Anzahl der zu erzeugenden I/Q-Samples.
    fs : float
        Abtastrate in Hz. Empfohlen >= 300 kHz, damit die
        Carson-Bandbreite (~256 kHz) sauber abgebildet wird.
    seed : int | None
        Startwert des Zufallsgenerators (fuer reproduzierbare Laeufe).
    return_mpx : bool
        Wenn True, wird zusaetzlich das reelle MPX-Basisbandsignal
        zurueckgegeben (nuetzlich zur Validierung der Empfangskette).

    Rueckgabe
    ---------
    iq : np.ndarray (complex128)
        Sendesignal, |iq| = 1, mittlere Leistung 1.
    mpx : np.ndarray (float64), nur wenn return_mpx=True
        Das erzeugte Stereo-Multiplex-Signal vor der Modulation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs

    # Zwei unabhaengige "Audiokanaele" (L+R und L-R)
    l_plus_r = _lowpass_noise(n_samples, AUDIO_CUTOFF_HZ, fs, rng)
    l_minus_r = _lowpass_noise(n_samples, AUDIO_CUTOFF_HZ, fs, rng)

    # Stereo-Multiplex zusammensetzen
    mpx = (0.45 * l_plus_r
           + 0.09 * np.sin(2 * np.pi * PILOT_HZ * t)
           + 0.45 * l_minus_r * np.sin(2 * np.pi * SUBCARRIER_HZ * t))

    # Auf Spitzenwert 1 normieren -> Spitzenhub entspricht DEVIATION_HZ
    mpx /= np.max(np.abs(mpx)) + 1e-12

    # FM-Modulation: Momentanphase = 2*pi*hub * Integral(mpx dt)
    phase = 2 * np.pi * DEVIATION_HZ * np.cumsum(mpx) / fs
    iq = np.exp(1j * phase)

    if return_mpx:
        return iq, mpx
    return iq
