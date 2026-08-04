"""
Empfaenger (Receiver) — Schritt 1: idealer Direktempfang.

In diesem ersten Schritt empfaengt die Antenne das Sendesignal *direkt*
und *ideal*: kein Ziel, keine Ausbreitungsverzoegerung, kein Doppler,
kein Rauschen. Der "Empfang" ist damit zunaechst eine 1:1-Abnahme des
I/Q-Signals — genau der Datenstrom, der in echter Hardware hinter dem
ADC anliegen wuerde (hier noch ohne Quantisierung).

Zur Kontrolle, dass die Kette wirklich funktioniert, enthaelt das Modul
einen FM-Demodulator: er gewinnt aus dem empfangenen I/Q-Signal das
MPX-Basisband zurueck. Stimmt dieses mit dem gesendeten MPX ueberein,
ist die Sender->Empfaenger-Kette korrekt.
"""

import numpy as np

from transmitter import DEVIATION_HZ


def receive(tx_iq):
    """Idealer Direktempfang: liefert das empfangene I/Q-Signal.

    Im idealen Fall ist das Empfangssignal identisch zum Sendesignal.
    Diese Funktion existiert als klarer Platzhalter fuer den Empfangs-
    schritt — hier werden spaeter Verzoegerung, Rauschen und ADC ergaenzt.
    """
    return tx_iq.copy()


def receive_array(s, a, dtype=np.complex64):
    """Direktpfad-Empfang an einem Antennenarray.

    Ebene Welle aus Richtung phi_tx: jedes Element sieht dasselbe Signal,
    nur mit eigener Phasenlage aus dem Steering-Vektor a:

        X_n(t) = a_n * s(t)

    Parameter
    ---------
    s : (M,) complex        Basisbandsignal des Senders
    a : (N,) complex        Steering-Vektor des Arrays
    Rueckgabe
    ---------
    X : (N, M) complex      Kanalmatrix, ein Zeile je Antennenelement

    Hier fehlen bewusst die gemeinsame Laufzeit tau = R/c und die
    Freiraumdaempfung des Direktpfads: beide wirken auf *alle* Kanaele
    gleich und aendern ohne Rauschen und ohne zweiten Pfad nichts an den
    Phasendifferenzen zwischen den Kanaelen. Sobald ein Zielecho dazu-
    kommt, gehoeren sie an genau diese Stelle — als gemeinsamer Faktor
    amp * s(t - tau), realisiert ueber eine (fraktionale) Verzoegerung.
    """
    return (a[:, None] * s[None, :]).astype(dtype)


def fm_demodulate(iq, fs):
    """FM-Demodulation: rekonstruiert das MPX-Basisband aus I/Q.

    Die Momentanfrequenz ist die Zeitableitung der Momentanphase.
    Diskret: f(t) = (1 / 2*pi) * d(phase)/dt. Durch Teilen durch den
    Frequenzhub erhaelt man wieder das (auf Spitzenwert ~1 normierte)
    MPX-Signal.

    Rueckgabe hat einen Wert weniger als die Eingabe (Differenzbildung);
    fuer Vergleiche einfach die Eingabe entsprechend kuerzen.
    """
    # Phasendifferenz aufeinanderfolgender Samples, robust entwickelt
    dphase = np.angle(iq[1:] * np.conj(iq[:-1]))
    inst_freq = dphase * fs / (2 * np.pi)
    return inst_freq / DEVIATION_HZ
