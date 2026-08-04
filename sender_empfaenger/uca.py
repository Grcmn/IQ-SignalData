"""
Antennengeometrie — uniformes Kreisarray (UCA) mit N Elementen.

Dieses Modul enthaelt ausschliesslich *Geometrie*: Elementpositionen,
Einfallsrichtung des Senders und den daraus folgenden Steering-Vektor.
Es kennt weder Signalform noch Abtastrate.

Modellannahme: Fernfeld (ebene Welle). Der Sender ist so weit entfernt,
dass die Wellenfront ueber die Arrayapertur hinweg als eben gilt. Damit
ist die Amplitude an allen Elementen gleich und nur die *Phase*
unterscheidet sich — genau das, was Richtungsschaetzung auswertet.

Konvention:
    - Element n liegt bei Winkel phi_n = 2*pi*n/N (n = 0 auf der +x-Achse,
      Zaehlung gegen den Uhrzeigersinn).
    - Elementposition: p_n = r * (cos(phi_n), sin(phi_n)).
"""

import numpy as np

C_LIGHT = 299_792_458.0     # Lichtgeschwindigkeit [m/s]
N_ELEMENTS = 7              # Anzahl Antennenelemente des UCA


def wavelength(fc):
    """Wellenlaenge lambda = c / fc [m]."""
    return C_LIGHT / fc


def radius_from_spacing(d, n=N_ELEMENTS):
    """Arrayradius r aus dem Abstand d benachbarter Elemente.

    Zwei benachbarte Elemente eines UCA spannen den Zentriwinkel 2*pi/N auf.
    Die Sehne zwischen ihnen hat die Laenge d = 2*r*sin(pi/N), also:

        r = d / (2 * sin(pi/N)).

    Ueblich ist d <= lambda/2, damit die Richtungsschaetzung eindeutig
    bleibt (keine Grating Lobes).
    """
    return d / (2.0 * np.sin(np.pi / n))


def element_angles(n=N_ELEMENTS):
    """Winkel phi_n der N Elemente auf dem Kreis [rad]."""
    return 2.0 * np.pi * np.arange(n) / n


def element_positions(r, n=N_ELEMENTS):
    """Elementpositionen als Array der Form (n, 2) in Metern."""
    phi = element_angles(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def azimuth(tx_pos, rx_pos=(0.0, 0.0)):
    """Azimut der einfallenden Welle am Arrayzentrum [rad].

    phi_tx = atan2(y_tx - y_rx, x_tx - x_rx), gemessen wie die
    Elementwinkel gegen den Uhrzeigersinn ab der +x-Achse.
    """
    return np.arctan2(tx_pos[1] - rx_pos[1], tx_pos[0] - rx_pos[0])


def element_phases(phi_tx, fc, r, n=N_ELEMENTS):
    """Phasenlagen alpha_n der ebenen Welle an den N Elementen [rad].

        alpha_n = (2*pi/lambda) * r * cos(phi_n - phi_tx)

    Herleitung: die Projektion der Elementposition p_n auf die
    Einfallsrichtung u = (cos phi_tx, sin phi_tx) betraegt
    p_n . u = r*cos(phi_n - phi_tx). Dieser Wegunterschied gegenueber dem
    Arrayzentrum entspricht der Phase 2*pi/lambda * (p_n . u).

    Positives Vorzeichen: Elemente, die dem Sender zugewandt sind, sehen
    die Wellenfront *frueher* (Phasenvorlauf).
    """
    lam = wavelength(fc)
    return (2.0 * np.pi / lam) * r * np.cos(element_angles(n) - phi_tx)


def steering_vector(phi_tx, fc, r, n=N_ELEMENTS):
    """Steering-Vektor a des UCA fuer die Einfallsrichtung phi_tx.

        a_n = exp(1j * alpha_n)

    Rueckgabe: komplexer Vektor der Laenge n, |a_n| = 1. Er haengt nur
    von der Senderrichtung ab, nicht von der Zeit — deshalb wird er einmal
    berechnet und auf jeden Signalblock angewendet.
    """
    return np.exp(1j * element_phases(phi_tx, fc, r, n))
