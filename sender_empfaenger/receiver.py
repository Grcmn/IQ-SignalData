"""
Empfaenger — vom Sendesignal zu den Daten der einzelnen Antennenkanaele.

Zwei Funktionen, zwei Richtungen:
    receive_array()  — Vorwaertsmodell: aus s(t) und dem Steering-Vektor
                       die Kanalmatrix X der N Antennenelemente bilden.
    fm_demodulate()  — Rueckwaerts zur Kontrolle: aus einem Kanal wieder
                       das modulierende m(t) gewinnen.

Diese Stufe modelliert ausschliesslich den *Direktpfad* Sender ->
Empfaenger im Fernfeld. Bewusst noch nicht enthalten: Laufzeit,
Daempfung, Rauschen, Zielecho, Doppler.
"""

import numpy as np

from transmitter import DEVIATION_HZ


def receive_array(s, a, dtype=np.complex64):
    """Kanalmatrix des Arrays fuer eine ebene Welle aus Richtung phi_tx.

    Jedes Element sieht dasselbe Signal, nur mit eigener Phasenlage aus
    dem Steering-Vektor a:

        X_n(t) = a_n * s(t)

    s : (M,) complex        Basisbandsignal des Senders
    a : (N,) complex        Steering-Vektor des Arrays
    Rueckgabe
    X : (N, M) complex      Kanalmatrix, eine Zeile je Antennenelement

    Hier fehlen bewusst die gemeinsame Laufzeit tau = R/c und die
    Freiraumdaempfung des Direktpfads: beide wirken auf *alle* Kanaele
    gleich und aendern ohne Rauschen und ohne zweiten Pfad nichts an den
    Phasendifferenzen zwischen den Kanaelen. Sobald ein Zielecho dazu-
    kommt, gehoeren sie an genau diese Stelle — als gemeinsamer Faktor
    amp * s(t - tau), realisiert ueber eine (fraktionale) Verzoegerung.
    """
    return (a[:, None] * s[None, :]).astype(dtype)


def fm_demodulate(iq, fs, deviation_hz=DEVIATION_HZ):
    """FM-Demodulation: rekonstruiert das modulierende m(t) aus I/Q.

    Die Momentanfrequenz ist die Zeitableitung der Momentanphase.
    Diskret: f(t) = (1 / 2*pi) * d(phase)/dt. Durch Teilen durch den
    Frequenzhub erhaelt man wieder das (auf Spitzenwert ~1 normierte)
    modulierende Signal.

    Eine konstante Phase des Eingangs (z. B. die Steering-Phase a_n eines
    Antennenelements) faellt bei der Differenzbildung heraus — m(t) kommt
    auf jedem Kanal identisch zurueck.

    Rueckgabe hat einen Wert weniger als die Eingabe (Differenzbildung);
    fuer Vergleiche einfach die Eingabe entsprechend kuerzen.
    """
    # Phasendifferenz aufeinanderfolgender Samples, robust entwickelt
    dphase = np.angle(iq[1:] * np.conj(iq[:-1]))
    inst_freq = dphase * fs / (2 * np.pi)
    return inst_freq / deviation_hz
