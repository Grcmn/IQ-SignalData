import numpy as np
from config import PILOT_HZ, DEVIATION_HZ, TARGET_RMS
from scipy.signal import bilinear, lfilter

def preemphasis(x, fs, tau=50e-6, f_pole=50e3):
    b_s = [tau, 1.0]
    a_s = [1.0 / (2.0 * np.pi * f_pole), 1.0]
    b, a = bilinear(b_s, a_s, fs)
    return lfilter(b, a, x)


def audio_lowpass(x, fs, f_cut=15e3, f_stop=19e3, a_stop=60.0):
    
    from scipy.signal import cheby2, cheb2ord, sosfilt
    order, wn = cheb2ord(f_cut, f_stop, 0.5, a_stop, fs=fs)
    sos = cheby2(order, a_stop, wn, btype="low", fs=fs, output="sos")
    return sosfilt(sos, x)


def agc_limit(left, right, target_rms=TARGET_RMS):
    # globale Normierung plus Soft-Limiter (pegel)
    g = target_rms / (max(np.std(left), np.std(right)) + 1e-12)
    return np.tanh(left * g), np.tanh(right * g)


class FMStream:

    def __init__(self, fs, left, right, deviation_hz=DEVIATION_HZ, pilot_hz=PILOT_HZ):

        self.fs = float(fs)
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)

        left = preemphasis(left, self.fs)
        right = preemphasis(right, self.fs)

        left, right = agc_limit(left, right)

        # Bandbegrenzung, entfernt die Clipping-Produkte
        left = audio_lowpass(left, self.fs)
        right = audio_lowpass(right, self.fs)

        self.left = np.clip(left, -1.0, 1.0)
        self.right = np.clip(right, -1.0, 1.0)

        self.deviation_hz = float(deviation_hz)
        self.pilot_hz = float(pilot_hz)

        self._n = 0         
        self._phase = 0.0   

    @property
    def n_samples(self):
        """Verfuegbare Samples insgesamt (= Laenge des Audiomaterials)."""
        return self.left.size

    @property
    def remaining(self):
        """Noch nicht ausgegebene Samples."""
        return self.n_samples - self._n

    def _multiplex(self, n0, count):
        t = (n0 + np.arange(count)) / self.fs #zeit wird aus sample index gebildet
        l = self.left[n0:n0 + count]
        r = self.right[n0:n0 + count]

        mono = 0.5 * (l + r)                               # 0 .. 15 kHz
        diff = 0.5 * (l - r)                               # -> DSB um 38 kHz
        subcarrier = np.sin(4.0 * np.pi * self.pilot_hz * t)
        pilot = np.sin(2.0 * np.pi * self.pilot_hz * t)

        return 0.9 * (mono + diff * subcarrier) + 0.1 * pilot

    def next_block(self, block_size):
        """Nächster Block des FM-Referenzsignals.

        Rueckgabe: (s, phase)
            s     (B,) komplex, e^(j*phase), Betrag 1
            phase (B,) reell, unwrapped Momentanphase

        Die Phase wird zusätzlich herausgegeben, weil der Echopfad sie
        braucht: verzögert wird phi, nicht s. Der Interpolator arbeitet
        damit auf einem stark 1/f-gewichteten Signal (Leistungsanteil
        oberhalb 0.45*Nyquist: 3e-06) statt auf dem FM-Signal selbst
        (4e-01), wo er nachweislich einbricht.

        Bei zu wenig verbleibendem Audiomaterial werden zwei leere Arrays
        zurückgegeben; der letzte unvollständige Block entfällt.
        """
        if self.remaining < block_size:
            return (np.empty(0, dtype=np.complex128),
                    np.empty(0, dtype=np.float64))

        m = self._multiplex(self._n, block_size)
        self._n += block_size

        inst_freq = self.deviation_hz * m                    # Δf * m(t) Momentanfreq (+-75)
        phase = self._phase + 2.0 * np.pi * np.cumsum(inst_freq) / self.fs # instantaneous phase ϕ(t)
        self._phase = float(phase[-1])   # Phasenübergabe an nächsten Block

        return np.exp(1j * phase), phase
