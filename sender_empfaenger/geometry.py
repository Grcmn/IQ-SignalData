"""Bistatische Geometrie.

Zustandslos und rein numerisch. Alle Funktionen arbeiten sowohl auf einer
einzelnen Position der Form (D,) als auch auf einem Positionsverlauf (T, D),
und sowohl in 2D als auch in 3D. Moeglich wird das durch durchgaengiges
axis=-1 bei den Normen: die letzte Achse ist immer die Raumachse, alle
davorliegenden Achsen sind Zeit.

Damit gibt es genau eine Implementierung fuer beide Aufrufarten, die im
Echopfad vorkommen:
  - pro Sample : p hat die Form (BLOCK_SIZE, 3) -> Traegerphase
  - pro Block  : p hat die Form (3,)            -> Verzoegerung, Winkel, Amplitude

"""

import numpy as np

from uca import C_LIGHT, wavelength


def _vec(p):
    """Wandelt Tupel/Liste/Array in ein float64-Array."""
    return np.asarray(p, dtype=np.float64)


def path_lengths(p, tx, rx):
    """
    p, tx, rx : Positionen, (D,) oder (T, D)
    Rückgabe : (r_t, r_r, L)
                r_t = Sender -> Ziel
                r_r = Ziel -> Empfaenger
                L   = Basislinie Sender -> Empfaenger (Skalar)
    """
    p, tx, rx = _vec(p), _vec(tx), _vec(rx)
    r_t = np.linalg.norm(p - tx, axis=-1)
    r_r = np.linalg.norm(p - rx, axis=-1)
    baseline = float(np.linalg.norm(tx - rx))
    return r_t, r_r, baseline


def bistatic_range(p, tx, rx):
    """Bistatische Entfernung R = r_t + r_r - L.  [M, Gl. 2.4]
    Ein Ziel auf der Basislinie liefert R = 0.
    """
    r_t, r_r, baseline = path_lengths(p, tx, rx)
    return r_t + r_r - baseline


def bistatic_velocity(p, v, tx, rx):
    """Bistatische Geschwindigkeit V = v.u_t + v.u_r.  [M, Gl. 2.6]

    Projektion des Geschwindigkeitsvektors auf die beiden Sichtlinien.
    V ist die zeitliche Ableitung von bistatic_range().

    Wird für die Signalerzeugung nicht gebraucht: die Trägerphase wird
    direkt aus r(t) gebildet, der Doppler entsteht dabei als Ableitung von
    selbst. V dient nur der Kontrollausgabe und dem Konsistenztest gegen die
    numerische Ableitung von r(t).
    """
    p, v = _vec(p), _vec(v)
    tx, rx = _vec(tx), _vec(rx)

    r_t, r_r, _ = path_lengths(p, tx, rx)
    u_t = (p - tx) / r_t[..., None]
    u_r = (p - rx) / r_r[..., None]

    return np.sum(v * (u_t + u_r), axis=-1)


def doppler_hz(v_bistatic, fc):
    """Bistatischer Doppler f_d = -V / lambda.  [M, Gl. 2.7]

    Vorzeichen: ein sich vom bistatischen Zentrum entfernendes Ziel (V > 0)
    erzeugt eine negative Dopplerfrequenz.
    """
    return -v_bistatic / wavelength(fc)


def delay_samples(r_bistatic, fs):
    """Bistatische Entfernung als Verzögerung in Samples.

    Bewusst ungerundet: der gebrochene Anteil ist die eigentliche Information
    und wird in target.VariableFractionalDelay per Interpolation umgesetzt.
    """
    return r_bistatic / C_LIGHT * fs


def echo_amplitude(p, tx, rx, rcs=1.0, loss_db=10.0):
    """Spannungsverhaeltnis Echo zu Direktpfad.

    Bistatische Radargleichung [M, Abschn. 2.3] geteilt durch Friis für den
    Direktpfad. Sendeleistung und Antennengewinne kürzen sich heraus, weil
    beide Pfade denselben Sender und dieselbe Empfangsantenne benutzen:

        P_echo / P_direkt = sigma * L^2 / (4*pi * r_t^2 * r_r^2 * L_sys)

    Rückgabe ist Wurzel davon, also ein Spannungs 
    weil der Wert direkt auf das Basisbandsignal multipliziert wird.

    Der Wert ist Reel. Der konstante Phasenanteil exp(j*2*pi*R/lambda), der
    im klassischen Modell [M, Gl. 4.7] Teil der komplexen Amplitude ist,
    steckt hier bereits im Trägerterm exp(j*2*pi*r(t)/lambda) bei t = 0 und
    darf nicht doppelt angebracht werden.

    Realistische Werte sind sehr klein: für sigma = 1 m^2 und das
    Beispielszenario rund -93 dB, bei int16-Ausgabe also unter einem LSB.
    Für Debuging über level_db in TargetEcho überschreibbar.
    """
    r_t, r_r, baseline = path_lengths(p, tx, rx)

    loss = 10.0 ** (loss_db / 10.0)
    power_ratio = (rcs * baseline ** 2 / (4.0 * np.pi * r_t ** 2 * r_r ** 2 * loss)) #loss und verhältnis aktuell nur annahmen

    return np.sqrt(power_ratio)
