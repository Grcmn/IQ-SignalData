"""Zielbahnen.

von geometry.py getrennt, leicht austauschbar oder erweiterbar 
für komplexere Zielbahn.

Zeitkonvention: t ist immer in Sekunden und bezieht sich auf t = 0 beim
ersten geschriebenen Sample.
"""

import numpy as np

class LinearTrajectory:
    """Geradlinig-gleichförmige Bewegung p(t) = p0 + v*t.
    Pro Ziel wird eine Trajektorie erzeugt.
    Die bistatische Beschleunigung entsteht daraus automatisch, weil die
    Umrechnung kartesisch -> bistatisch nichtlinear ist [M, Abschn. 9.4].
    Sie muss also nicht eigens modelliert werden - solange r(t) exakt aus
    p(t) berechnet wird und nicht als r ~ R + V*t linearisiert.
    """

    def __init__(self, p0, v):
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        if self.p0.shape != self.v.shape:
            raise ValueError("p0 und v muessen dieselbe Dimension haben")

    @property
    def n_dim(self):
        return self.p0.shape[-1]

    def position(self, t):
        """Position zur Zeit t.

        t skalar     -> (D,)
        t Form (T,)  -> (T, D)

        Beide Formen werden gebraucht: (T, D) fuer die Trägerphase pro
        Sample, (D,) fuer die blockweise Auswertung zur Blockmitte.
        """
        t = np.asarray(t, dtype=np.float64)
        return self.p0 + t[..., None] * self.v

    def velocity(self, t):
        """Geschwindigkeit zur Zeit t."""
        t = np.asarray(t, dtype=np.float64)
        return np.broadcast_to(self.v, t.shape + (self.n_dim,))
