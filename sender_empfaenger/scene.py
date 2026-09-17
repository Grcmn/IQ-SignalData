"""Zusammensetzen des Empfangssignals aus Direktpfad und Zielen.
Nach [M, Gl. 5.2] besteht das Signal am n-ten Antennenelement aus

    X_n(t) = DPI + Clutter + Ziele + Rauschen

Umgesetzt sind aktuell DPI und Ziele. Clutter ist physikalisch identisch zu
einem Ziel, nur mit v = 0. In dict eintragen wenn gewünscht.

Der Direktpfad ist eine Punktquelle mit R = 0 und f_d = 0 [M, Gl. 5.2]:
Amplitude 1, Steering-Vektor in Senderrichtung. 
Alle Echopegel sind relativ dazu definiert.
"""

import numpy as np

from receiver import receive_array
from target import IntegerDelay, TargetEcho
from trajectory import LinearTrajectory
from uca import look_angles, steering_vector


class Scene:
    """Hält Direktpfad und alle Ziele und liefert pro Block das Arraysignal.
    Alle Ziele bekommen dasselbe n_taps und damit dieselbe Gruppenlaufzeit M.
    """

    def __init__(self, targets, tx, rx, fc, fs, array_radius, n_elements,
                 duration_s, n_taps=33, loss_db=10.0,
                 static_amplitude=False):
        self.n_elements = int(n_elements)

        # Direktpfad: konstant, weil Sender und Empfänger stillstehen.
        phi_tx, theta_tx = look_angles(tx, rx)
        self.a_direct = steering_vector(float(phi_tx), fc, array_radius, n_elements, float(theta_tx))

        self.targets = [
            TargetEcho(
                trajectory=LinearTrajectory(cfg["pos"], cfg["vel"]),
                tx=tx, rx=rx, fc=fc, fs=fs,
                array_radius=array_radius, n_elements=n_elements,
                duration_s=duration_s,
                rcs=cfg.get("rcs", 1.0),
                level_db=cfg.get("level_db"),
                loss_db=loss_db,
                n_taps=n_taps,
                static_amplitude=static_amplitude,
                name=cfg.get("name", f"target{i}"),
            )
            for i, cfg in enumerate(targets)
        ]

        # Gemeinsame Latenz. Ohne Ziele bleibt sie 0.
        self.group_delay = max((t.group_delay for t in self.targets), default=0)
        self._ref_delay = IntegerDelay(self.group_delay, dtype=np.complex128)

    @property
    def peak_amplitude(self):
        """Obere Schranke fuer |X_n|, gebraucht zur Aussteuerung.

        Da |a_n| = 1 und |s| = 1 gilt, addieren sich Direktpfad und Echos im
        ungueenstigsten Fall konstruktiv zu 1 + Summe der Echoamplituden.
        """
        amps = [t._amplitude(t.traj.position(0.0)) for t in self.targets]
        return 1.0 + float(np.sum(amps)) if amps else 1.0

    def process(self, s, phase, n0):
        """Arraysignal eines Blocks.

        s     : (B,) komplexes FM-Referenzsignal
        phase : (B,) zugehörige Momentanphase, Eingang der Echopfade
        n0    : absoluter Sampleindex des Blockanfangs

        Rueckgabe: (n_elements, B) komplex
        """
        x = receive_array(self._ref_delay.process(s), self.a_direct, dtype=np.complex128) # direktpfad an allen Antennen

        for tgt in self.targets:
            x_e, a_tgt = tgt.process(phase, n0) # echo und steervect zur blockmitte
            x += receive_array(x_e, a_tgt, dtype=np.complex128) #zsmaddieren

        return x
