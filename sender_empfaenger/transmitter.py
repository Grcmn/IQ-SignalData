"""
Sender (Transmitter) — Erzeugung des FM-Sendesignals.

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

FMStream ist die *einzige* Signalquelle des Projekts: sie erzeugt das
Basisband blockweise mit fortgefuehrtem Zustand, damit ein kontinuier-
licher Datenstrom ohne Spruenge an den Blockgrenzen entsteht. Zwei Dinge
muessen dafuer ueber die Bloecke hinweg fortgefuehrt werden:
    1. die Momentanphase der FM (Integrationszustand), und
    2. die Zeitachse des modulierenden Signals m(t).
Beides erledigt FMStream ueber einen laufenden Sample-Zaehler und eine
gespeicherte Restphase.
"""

import numpy as np

# Kennwerte des UKW-Rundfunks
PILOT_HZ = 19e3            # Stereo-Pilotton
SUBCARRIER_HZ = 38e3       # Traeger des Stereo-Differenzsignals (2 x Pilot)
AUDIO_CUTOFF_HZ = 15e3     # obere Audiogrenzfrequenz
DEVIATION_HZ = 75e3        # maximaler Frequenzhub

# Voreingestellte Toene fuer das modulierende Signal m(t):
# (Frequenz [Hz], relative Amplitude). Die Summe der Amplituden dient als
# Normierung -> m(t) liegt garantiert in [-1, 1], und zwar *blockunabhaengig*
# (eine blockweise Maximum-Normierung wuerde an jeder Blockgrenze springen).
DEFAULT_TONES = ((440.0, 1.0), (1_000.0, 0.6), (3_500.0, 0.3), (11_000.0, 0.15))


class FMStream:
    """Blockweise Quelle eines FM-Basisbandsignals s(t) = exp(1j*phase(t)).

    Parameter
    ---------
    fs : float
        Abtastrate in Hz.
    deviation_hz : float
        Frequenzhub delta_f; Momentanfrequenz f(t) = delta_f * m(t).
    mode : {"tones", "mpx"}
        "tones" — m(t) ist eine Summe weniger Sinustoene (einfach,
                  reproduzierbar, gut zum Nachrechnen der Geometrie).
        "mpx"   — m(t) hat die Struktur eines Stereo-Multiplex mit
                  19-kHz-Pilot und Differenzsignal um 38 kHz; naeher am
                  echten UKW-Rundfunk.
    tones : Sequenz von (frequenz_hz, amplitude)
        Toene des Audioanteils.

    Der erzeugte Betrag ist konstant 1 (reine FM), die mittlere Leistung
    damit ebenfalls 1.
    """

    def __init__(self, fs, deviation_hz=DEVIATION_HZ, mode="mpx",
                 tones=DEFAULT_TONES):
        if mode not in ("tones", "mpx"):
            raise ValueError("mode muss 'tones' oder 'mpx' sein")
        self.fs = float(fs)
        self.deviation_hz = float(deviation_hz)
        self.mode = mode
        self.tones = tuple(tones)
        self._n = 0          # laufender Sample-Index (Zeitachse von m(t))
        self._phase = 0.0    # Restphase am Ende des letzten Blocks [rad]

    # -- modulierendes Signal m(t), normiert auf [-1, 1] ---------------
    def _modulation(self, t):
        audio = sum(a * np.sin(2 * np.pi * f * t) for f, a in self.tones)
        norm = sum(abs(a) for _, a in self.tones)
        audio = audio / norm                      # -> [-1, 1]

        if self.mode == "tones":
            return audio

        # MPX: Mono-Summe + Pilot + Stereo-Differenz als DSB um 38 kHz.
        # Als "L-R" dient hier eine phasenverschobene Variante des Audios.
        diff = sum(a * np.cos(2 * np.pi * f * t) for f, a in self.tones) / norm
        mpx = (0.45 * audio
               + 0.09 * np.sin(2 * np.pi * PILOT_HZ * t)
               + 0.45 * diff * np.sin(2 * np.pi * SUBCARRIER_HZ * t))
        return mpx / 0.99     # feste Skalierung, Spitzenwert bleibt <= 1

    def next_block(self, block_size):
        """Liefert die naechsten block_size Samples als complex128."""
        # Absolute Zeitachse -> m(t) laeuft nahtlos weiter
        t = (self._n + np.arange(block_size)) / self.fs
        self._n += block_size

        m = self._modulation(t)
        inst_freq = self.deviation_hz * m                 # f(t) [Hz]

        # Phasenintegration; self._phase traegt den Zustand des Vorblocks
        phase = self._phase + 2 * np.pi * np.cumsum(inst_freq) / self.fs
        self._phase = float(phase[-1])                    # Zustand sichern

        return np.exp(1j * phase)

