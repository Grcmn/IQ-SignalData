"""
Analyse-Hilfsfunktionen zur Validierung der erzeugten I/Q-Daten.

Diese Funktionen gehören konzeptionell zur nachgelagerten
Signalverarbeitung (nicht zum Generator) und dienen hier nur dazu,
die generierten Szenarien gegen die Ground Truth zu prüfen.
"""

import numpy as np


def batch_caf(ref_sig: np.ndarray, surv_sig: np.ndarray, fs: float,
              n_delay: int, batch_len: int):
    """
    Schnelle Kreuzambiguitätsfunktion (Range-Doppler-Map) über das
    Batch-Verfahren: Signal in K Blöcke der Länge batch_len teilen,
    pro Block zirkulare Kreuzkorrelation (per FFT), dann FFT über die
    Blockachse -> Dopplerachse.

    Gültig für Delays << batch_len (SNR-Verlust ~ delay/batch_len);
    Doppler eindeutig bis ±fs/(2*batch_len), Auflösung fs/(K*batch_len).

    Rückgabe: (CAF [Doppler x Delay], Dopplerachse [Hz])
    """
    K = len(ref_sig) // batch_len
    r = ref_sig[:K * batch_len].reshape(K, batch_len)
    s = surv_sig[:K * batch_len].reshape(K, batch_len)
    corr = np.fft.ifft(np.fft.fft(s, axis=1) * np.conj(np.fft.fft(r, axis=1)),
                       axis=1)[:, :n_delay]
    caf = np.fft.fftshift(np.fft.fft(corr, axis=0), axes=0)
    doppler = np.fft.fftshift(np.fft.fftfreq(K, batch_len / fs))
    return caf, doppler
