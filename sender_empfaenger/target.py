"""Echopfad: Verzoegerungsleitungen und Zielecho.

Umgesetzt wird [M, Gl. 4.24] ohne die dort anschliessend eingefuehrten
Vereinfachungen [M, Gl. 4.5/4.6]:

    x_e(t) = C(t) * x_r(t - r(t)/c) * exp(j*2*pi*r(t)/lambda)

mit r(t) exakt aus der Zielposition, nicht als r ~ R + V*t linearisiert.

Zwei Update-Raten, weil die beteiligten Größen um Größenordnungen
verschieden empfindlich sind:

  Trägerphase   Maßstab lambda = 3.35 m   -> pro Sample
  Verzögerung   Maßstab dR = c/B ~ 2 km   -> pro Block
  Steering       Maßstab lambda/(2*pi*r)   -> pro Block
  Amplitude      Maßstab ~0.5 dB           -> pro Block

Die Trägerphase muss samplegenau sein, weil sie ein Ton mit der
Dopplerfrequenz ist (hier rund 52 Hz). Die Blockrate beträgt nur
fs/BLOCK_SIZE = 37 Hz - blockweise gehalten würde der Doppler
unterabgetastet und der CAF-Peak säße an der falschen Stelle.
"""

import numpy as np

from geometry import bistatic_range, delay_samples, echo_amplitude
from uca import look_angles, steering_vector, wavelength


class IntegerDelay:
    """Verzögerung um genau d Samples, ohne Interpolation.

    Zweck ist ausschliesslich der Latenzausgleich: der Interpolator im
    Echopfad hat die Gruppenlaufzeit M = (n_taps-1)/2. Diese M Samples sind
    reine Implementierungslatenz ohne physikalische Bedeutung. Damit die
    RELATIVE Verzögerung zwischen Referenz und Echo stimmt muss der 
    Referenzpfad um dasselbe M verzögert werden.
    y[n] = x[n - 16]
    """

    def __init__(self, d, dtype=np.complex128):
        self.d = int(d)
        self._hist = np.zeros(self.d, dtype=dtype)

    def process(self, x):
        """Gibt genauso viele Samples zurück, wie hineingegeben wurden."""
        if self.d == 0:
            return np.array(x, copy=True)
        buf = np.concatenate([self._hist, x])
        self._hist = buf[-self.d:].copy()
        return buf[:len(x)]


class VariableFractionalDelay:
    """Verzögerung um einen nicht-ganzzahligen Betrag, pro Block nachführbar.

    Die Zielentfernung ergibt keine ganze Sampleanzahl - im Beispielszenario
    2.2727 Samples. Ein Runden auf 2 entspraeche 538 m Entfernungsfehler.
    Der Zwischenwert wird per Sinc-Interpolation rekonstruiert; das
    Abtasttheorem garantiert, dass das fuer bandbegrenzte Signale exakt
    moeglich ist, die Kuerzung auf n_taps erzeugt den Restfehler.

    Aufteilung:
      ganzzahliger Anteil d_int : reiner Pufferzugriff, fehlerfrei
      gebrochener Anteil  frac  : gefensterter Sinc-Kern

    Der Kern verzögert nicht um frac, sondern um M + frac, weil er um M
    zentriert ist. Diese Latenz M wird auf dem Referenzpfad mit IntegerDelay
    ausgeglichen.

    set_delay() tauscht NUR den Kern aus. Die History bleibt unangetastet -
    sie wird beim Konstruieren einmal auf d_max ausgelegt. Eine Neuallokation
    zur Laufzeit wuerde den Zustand zerstören und wäre an jeder Blockgrenze
    als Sprung sichtbar.

    """

    def __init__(self, d_max, n_taps=33, dtype=np.float64):
        if n_taps % 2 == 0:
            raise ValueError("n_taps muss ungerade sein (symmetrischer Kern)")

        self.n_taps = int(n_taps)
        self.group_delay = (self.n_taps - 1) // 2       # = M

        # +1 Reserve, damit floor(d) == d_int_max noch passt
        self._d_int_max = int(np.floor(d_max)) + 1

        # So viele vergangene Eingangssamples braucht das aelteste
        # Ausgangssample eines Blocks.
        self._hist_len = self._d_int_max + self.n_taps - 1
        self._hist = np.zeros(self._hist_len, dtype=dtype)

        self._d_int = 0
        self._h = None
        self.set_delay(0.0)

    def set_delay(self, d):
        """Baut einen Filterkern welcher das Signal um frac verschiebt."""
        d_int = int(np.floor(d))
        frac = d - d_int

        if not 0 <= d_int <= self._d_int_max:
            raise ValueError(
                f"Verzoegerung {d:.3f} liegt ausserhalb des beim Konstruieren "
                f"reservierten Bereichs 0..{self._d_int_max}")

        k = np.arange(self.n_taps)
        h = np.sinc(k - self.group_delay - frac) * np.hanning(self.n_taps)

        self._h = h / h.sum()          # DC-Verstaerkung exakt 1
        self._d_int = d_int

    def process(self, x):
        """Verzoegert einen Block und gibt gleich viele Samples zurueck.

        Der Ausgang trägt zusätzlich die Gruppenlaufzeit M.
        """
        buf = np.concatenate([self._hist, x])
        self._hist = buf[-self._hist_len:].copy()

        y = np.convolve(buf, self._h) # durch faltung den berechneten Filterkern über buf schieben

        # Herleitung des Offsets:
        #   buf[hist_len + i] entspricht x[i]
        #   y[m] entspricht buf(m - M - frac)
        #   gesucht ist out[i] = x(i - d_int - frac - M)
        #   -> m = hist_len + i - d_int
        start = self._hist_len - self._d_int
        return y[start:start + len(x)]


class TargetEcho:
    """Erzeugt das Echosignal eines bewegten Punktziels.

    Eingang ist die MOMENTANPHASE des FM-Referenzsignals, nicht das Signal
    selbst. Grund: verzoegert werden muss die Einhuellende, und dafuer ist
    die Phase der mit Abstand guenstigste Traeger der Information.

        s(t) = exp(j*phi(t))   =>   s(t - tau) = exp(j*phi(t - tau))

    Der Interpolator arbeitet also auf phi statt auf s. Gemessener
    Leistungsanteil oberhalb 0.45*Nyquist, wo der Interpolator einbricht:

        FM-Signal s : 4.1e-01
        MPX m       : 2.0e-01
        Phase phi   : 3.2e-06

    Das sind fuenf Groessenordnungen. phi ist das Integral ueber m und damit
    stark 1/f-gewichtet, liegt also praktisch vollstaendig im Bereich, in dem
    der Interpolator nachweislich fehlerfrei ist. Zusaetzlich entfaellt ein
    zweiter Phasenakkumulator und dessen Offsetdrift.

    Das ist eine Synthese-Abkuerzung: sie ist nur zulaessig, weil das
    Sendesignal erzeugt und nicht gemessen wird. Ein reales System haette
    keinen Zugriff auf phi.
    """

    def __init__(self, trajectory, tx, rx, fc, fs, array_radius,
                 n_elements, duration_s, rcs=1.0, level_db=None,
                 loss_db=10.0, n_taps=33, static_amplitude=False,
                 name="target"):
        self.traj = trajectory
        self.tx = np.asarray(tx, dtype=np.float64)
        self.rx = np.asarray(rx, dtype=np.float64)
        self.fc = float(fc)
        self.fs = float(fs)
        self.lam = wavelength(fc)
        self.array_radius = float(array_radius)
        self.n_elements = int(n_elements)
        self.rcs = float(rcs)
        self.level_db = level_db
        self.loss_db = float(loss_db)
        self.static_amplitude = bool(static_amplitude)
        self.name = name

        # ÄNDERN , Pauschal einplanen
        # Maximale Verzoegerung ueber die gesamte Dateidauer vorab bestimmen.
        # Die History der Verzoegerungsleitung muss darauf ausgelegt sein,
        # weil sie zur Laufzeit nicht mehr veraendert werden darf.
        t_probe = np.linspace(0.0, duration_s, 256)
        r_probe = bistatic_range(self.traj.position(t_probe), self.tx, self.rx)
        d_max = delay_samples(r_probe.max(), self.fs)

        self._delay = VariableFractionalDelay(d_max, n_taps=n_taps,
                                              dtype=np.float64)

        # Feste Amplitude, falls der physikalische Verlauf nicht gewünscht
        # ist (für CAF-Vergleiche mit konstanter Peakhöhe).
        self._amp_fixed = (self._amplitude(self.traj.position(0.0))
                           if self.static_amplitude else None)

    @property
    def group_delay(self):
        """Latenz M des Interpolators, die der Referenzpfad ausgleichen muss."""
        return self._delay.group_delay

    def _amplitude(self, p):
        """Reelle Echoamplitude an der Position p.

        level_db = None bedeutet: physikalischer Wert aus der bistatischen
        Radargleichung. Ein gesetzter Wert überschreibt ihn - nötig fuer die
        Entwicklung, weil der physikalische Pegel (rund -93 dB) bei
        int16-Ausgabe unter einem LSB liegt.
        """
        if self.level_db is not None:
            return 10.0 ** (self.level_db / 20.0)
        return float(echo_amplitude(p, self.tx, self.rx, self.rcs, self.loss_db))

    def process(self, phase, n0):
        """Echo eines Blocks.

        phase : (B,) Momentanphase des FM-Referenzsignals, unwrapped
        n0    : absoluter Sampleindex des Blockanfangs

        n0 ist die einzige Uhr des Systems. In der .dat steht kein
        Zeitstempel; alles, was ueber Zeit bekannt ist, ist t = n/fs. Ein
        blocklokaler Index wuerde die Trajektorie bei jedem Block auf t = 0
        zuruecksetzen - das Ziel wuerde 372 mal von vorne losfliegen.

        Rueckgabe: (x_e, a_tgt)
            x_e   (B,) komplex, Skalarsignal des Echos
            a_tgt (n_elements,) Steering-Vektor zur Blockmitte

        Nicht das fertige (n, B)-Array, damit receiver.py unverändert bleibt
        und mehrere Ziele sich einfach aufsummieren lassen.
        """
        n_block = len(phase)
        M = self.group_delay

        # absolute Zeit jedes Ausgangs-Samples, korrigiert um die Interpolatorlatenz.
        t = (n0 + np.arange(n_block) - M) / self.fs

        # --- pro Sample: Entfernung für Trägerphase --------------------------------------
        p = self.traj.position(t)                       # (B, D)
        r = bistatic_range(p, self.tx, self.rx)         # (B,)

        # --- pro Block (ausgewertet zur Blockmitte) ----------
        # Blockmitte statt Blockanfang halbiert den systematischen Fehler.
        t_mid = float(t[n_block // 2])
        p_mid = self.traj.position(t_mid)               # (D,)

        r_mid = float(bistatic_range(p_mid, self.tx, self.rx))
        self._delay.set_delay(delay_samples(r_mid, self.fs))

        phi, theta = look_angles(p_mid, self.rx)
        a_tgt = steering_vector(float(phi), self.fc, self.array_radius, self.n_elements, float(theta))

        amp = (self._amp_fixed if self._amp_fixed is not None
               else self._amplitude(p_mid))

        # --- Zusammensetzen -------------------------------------------------
        # Reihenfolge: erst verzögern, dann Trägerphase.

        phase_delayed = self._delay.process(phase)
        s_delayed = np.exp(1j * phase_delayed) # |s_delayed| = 1 / x_r(t − r(t)/c)

        # amp ist reell: der konstante Phasenanteil exp(j*2*pi*R/lambda) aus
        # [M, Gl. 4.7] steckt bereits in exp(j*2*pi*r(t)/lambda) bei t = 0.
        
        x_e = amp * s_delayed * np.exp(2j * np.pi * r / self.lam)

        return x_e, a_tgt
