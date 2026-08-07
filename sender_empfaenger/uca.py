
import numpy as np

C_LIGHT = 299_792_458.0     
N_ELEMENTS = 7              


def wavelength(fc):
    return C_LIGHT / fc


def radius_from_spacing(d, n=N_ELEMENTS):
    # r = d / (2 * sin(pi/N)).
    return d / (2.0 * np.sin(np.pi / n))


def element_angles(n=N_ELEMENTS):
    return 2.0 * np.pi * np.arange(n) / n


def element_positions(r, n=N_ELEMENTS):
    phi = element_angles(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def azimuth(tx_pos, rx_pos=(0.0, 0.0)):
    return np.arctan2(tx_pos[1] - rx_pos[1], tx_pos[0] - rx_pos[0])


def element_phases(phi_tx, fc, r, n=N_ELEMENTS):
    # alpha_n = (2*pi/lambda) * r * cos(phi_n - phi_tx)
    lam = wavelength(fc)
    return (2.0 * np.pi / lam) * r * np.cos(element_angles(n) - phi_tx)


def steering_vector(phi_tx, fc, r, n=N_ELEMENTS):
    # a_n = exp(1j * alpha_n)
    return np.exp(1j * element_phases(phi_tx, fc, r, n))
