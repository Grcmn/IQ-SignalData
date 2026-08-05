"""
IQ-Datenstrom eines FM-Senders an einem 7-Element-UCA.

Erzeugt blockweise die Empfangsdaten aller sieben Antennenelemente fuer
den *Direktpfad* Sender -> Empfaenger. Bewusst noch nicht enthalten:
Rauschen, Zielecho, Doppler, Laufzeit.

Kette:
    FM-Basisband s(t)          (transmitter.FMStream, blockweise)
      x  Steering-Vektor a     (uca.steering_vector, aus der Geometrie)
      -> Kanalmatrix (7, block_size)

Dieses Modul enthaelt nur die Signalerzeugung — keine Ausgaben, keine
Grafik, kein matplotlib-Import. Der Einstiegspunkt ist run.py, die
Kontrollgrafiken stehen in plots.py.
"""

import numpy as np

from uca import (N_ELEMENTS, azimuth, element_phases, radius_from_spacing,
                 steering_vector, wavelength)
from receiver import receive_array
from transmitter import DEVIATION_HZ, FMStream

# --- Parameter des Szenarios -----------------------------------------
# Einzige Stelle, an der die Kenngroessen des Streams festgelegt werden.
FC_HZ = 100e6           # Traegerfrequenz des UKW-Senders [Hz]
FS_HZ = 240e3           # Abtastrate des Basisbands [Hz]
N = N_ELEMENTS          # Anzahl Antennenelemente (7)
D_OVER_LAMBDA = 0.4     # Elementabstand d als Vielfaches von lambda
ARRAY_RADIUS_M = None   # r direkt vorgeben; None -> aus d berechnen
TX_POS_M = (5_000.0, 3_000.0)   # Senderposition [m]
RX_POS_M = (0.0, 0.0)           # Arrayzentrum des Empfaengers [m]
# DEVIATION_HZ wird aus transmitter.py uebernommen (dort steht die Norm
# des UKW-Rundfunks: 75 kHz) und hier nur re-exportiert, damit run.py alle
# Szenarioparameter aus einem Modul beziehen kann.
BLOCK_SIZE = 4096       # Samples je Streaming-Block
N_BLOCKS = 20           # Anzahl Bloecke der Demo
MODE = "mpx"            # Modulationsquelle: "tones" oder "mpx"


def fm_uca_stream(fc=FC_HZ, fs=FS_HZ, n=N, d_over_lambda=D_OVER_LAMBDA,
                  r=ARRAY_RADIUS_M, tx_pos=TX_POS_M, rx_pos=RX_POS_M,
                  deviation_hz=DEVIATION_HZ, block_size=BLOCK_SIZE,
                  n_blocks=None, mode=MODE):
    """Generator: liefert Bloecke der Form (n, block_size), dtype complex64.

    Parameter
    ---------
    fc, fs : float          Traegerfrequenz und Abtastrate [Hz]
    n : int                 Anzahl Elemente des Kreisarrays
    d_over_lambda : float   Elementabstand d = d_over_lambda * lambda
    r : float | None        Arrayradius [m]; None -> aus d berechnet
    tx_pos, rx_pos : (x, y) Positionen in Metern
    deviation_hz : float    Frequenzhub der FM
    block_size : int        Samples je Block
    n_blocks : int | None   Anzahl Bloecke; None -> unendlicher Strom
    mode : str              Modulationsquelle, siehe transmitter.FMStream

    Verwendung:
        for block in fm_uca_stream(n_blocks=20):
            ...  # block.shape == (7, 4096)
    """
    # -- Geometrie (einmalig, zeitunabhaengig) -------------------------
    lam = wavelength(fc)
    if r is None:
        r = radius_from_spacing(d_over_lambda * lam, n)
    phi_tx = azimuth(tx_pos, rx_pos)
    a = steering_vector(phi_tx, fc, r, n)

    # -- Signalquelle mit blockuebergreifendem Phasenzustand -----------
    src = FMStream(fs, deviation_hz=deviation_hz, mode=mode)

    block = 0
    while n_blocks is None or block < n_blocks:
        s = src.next_block(block_size)          # (block_size,) Basisband
        yield receive_array(s, a)               # (n, block_size) complex64
        block += 1


def stream_info(fc=FC_HZ, fs=FS_HZ, n=N, d_over_lambda=D_OVER_LAMBDA,
                r=ARRAY_RADIUS_M, tx_pos=TX_POS_M, rx_pos=RX_POS_M):
    """Geometriegroessen des Streams (lambda, r, phi_tx, alpha, a, n)."""
    lam = wavelength(fc)
    if r is None:
        r = radius_from_spacing(d_over_lambda * lam, n)
    phi_tx = azimuth(tx_pos, rx_pos)
    alpha = element_phases(phi_tx, fc, r, n)
    return dict(lam=lam, r=r, n=n, phi_tx=phi_tx, alpha=alpha,
                a=np.exp(1j * alpha))
