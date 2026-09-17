"""Uniform Circular Array (UCA) in der xyz-Ebene.
"""

import numpy as np

C_LIGHT = 299_792_458.0
N_ELEMENTS = 7                       # 7 physische Antennenelemente


def wavelength(fc): 
    return C_LIGHT / fc #λ


def radius_from_spacing(d, n=N_ELEMENTS):
    """Arrayradius aus Elementabstand d auf dem Kreis."""
    return d / (2.0 * np.sin(np.pi / n))


def element_angles(n=N_ELEMENTS): #weiß matlab was das 0te element ist?
    """Winkelpositionen der Elemente auf dem Kreis, gleichverteilt."""
    return 2.0 * np.pi * np.arange(n) / n


def look_angles(p, rx):
    """Azimut und Elevation einer Position, gesehen vom Empfänger.
    """
    p = np.asarray(p, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)

    # rx auf dieselbe Dimensionszahl bringen
    # d ist vektor von empfänger zu punkt
    d = p - rx[..., :p.shape[-1]]

    phi = np.arctan2(d[..., 1], d[..., 0]) # azimut
    rho = np.hypot(d[..., 0], d[..., 1]) # √(dx² + dy²)
    theta = np.arctan2(rho, d[..., 2]) # winkel zw richtung und z achse

    return phi, theta


def azimuth(p, rx=(0.0, 0.0, 0.0)):
    """Nur der Azimut"""
    return look_angles(p, rx)[0]


def element_phases(phi, fc, r, n=N_ELEMENTS, theta=0.5 * np.pi):
    """Phasenversatz der n Elemente gegenueber dem Arrayzentrum.

        alpha_n = (2*pi/lambda) * r * sin(theta) * cos(phi_n - phi)

    Der Faktor sin(theta) ist die Projektion der Einfallsrichtung auf die Arrayebene [Kap 4.7]. 
    Rückgabe: (7,)
    """
    lam = wavelength(fc)
    return ((2.0 * np.pi / lam) * r * np.sin(theta) * np.cos(element_angles(n) - phi))


def steering_vector(phi, fc, r, n=N_ELEMENTS, theta=0.5 * np.pi):
    """Steering-Vektor a_n = exp(j*alpha_n), Betrag 1 je Element.
    Trägt die gesamte Winkelinformation. Direktpfad und Echo bekommen jeweils ihren eigenen.
    """
    return np.exp(1j * element_phases(phi, fc, r, n, theta))
