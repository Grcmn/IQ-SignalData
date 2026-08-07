
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def load_audio(path, fs, duration_s=None):
    """Stereo-Audio aus einer Datei auf die Abtastrate fs bringen.

    Parameter
    ---------
    path : str | None
        Pfad zur Audiodatei (MP3, WAV, FLAC, ...). None -> Platzhalter,
        siehe placeholder_audio().
    fs : float
        Zielabtastrate [Hz] (= Abtastrate des Basisbands).
    duration_s : float | None
        Auf diese Laenge kuerzen; None -> ganze Datei.

    Rueckgabe
    ---------
    (left, right) : je (M,) float64, Wertebereich [-1, 1]
    """
    if path is None:
        return placeholder_audio(fs, duration_s or 1.0)

    import soundfile as sf                       # nur hier benoetigt

    data, fs_file = sf.read(path, dtype="float64", always_2d=True)

    # Mono -> beide Kanaele gleich; mehr als 2 Kanaele -> die ersten zwei
    left = data[:, 0]
    right = data[:, 1] if data.shape[1] >= 2 else data[:, 0]

    left = _resample(left, fs_file, fs)
    right = _resample(right, fs_file, fs)

    if duration_s is not None:
        m = int(round(duration_s * fs))
        left, right = left[:m], right[:m]

    return _normalize(left, right)


def placeholder_audio(fs, duration_s=1.0, seed=0):
    """PLATZHALTER, bis eine MP3 vorliegt.

    Erzeugt bandbegrenztes Rauschen (0 .. 15 kHz) mit leicht
    unterschiedlichem Inhalt auf L und R, damit das Differenzsignal L-R
    nicht verschwindet und der Stereoanteil des Multiplex wirksam wird.

    Bandbegrenztes Rauschen ist ein bewusst gewaehlter Platzhalter: es ist
    das uebliche Modell fuer Illuminatorsignale in der Passivradarliteratur
    [Malanowski 2019, Abschn. 2.4.1 und 4.2] und liefert eine volle,
    zeitlich konstante Audiobandbreite -- also den *guenstigsten* Fall.
    Echtes Programmmaterial ist schmaler und schwankt.
    """
    rng = np.random.default_rng(seed)
    m = int(round(duration_s * fs))

    # Weisses Rauschen erzeugen und im Frequenzbereich auf 15 kHz begrenzen
    def band_noise():
        x = rng.standard_normal(m)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(m, 1.0 / fs)
        X[f > 15e3] = 0.0
        return np.fft.irfft(X, n=m)

    return _normalize(band_noise(), band_noise())


def _resample(x, fs_in, fs_out):
    """Abtastratenwandlung ueber ein exaktes rationales Verhaeltnis.

    fs_out / fs_in wird als Bruch p/q dargestellt (z.B. 240000/44100 =
    800/147); resample_poly interpoliert um p und dezimiert um q, mit
    integriertem Antialiasing-Filter. Dadurch ist fs im Parameterblock frei
    waehlbar, ohne dass hier etwas angepasst werden muss.
    """
    if int(fs_in) == int(fs_out):
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(int(fs_out), int(fs_in))
    return resample_poly(x, ratio.numerator, ratio.denominator)


def _normalize(left, right):
    """Gemeinsame Skalierung beider Kanaele auf Spitzenwert 1.

    Gemeinsam, nicht je Kanal: sonst wuerde das Stereobild verzerrt und
    das Differenzsignal L-R kuenstlich veraendert.
    """
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak == 0.0:
        return left, right
    return left / peak, right / peak