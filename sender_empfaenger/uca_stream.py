"""
Schritt 2: IQ-Datenstrom eines FM-Senders an einem 7-Element-UCA.

Erzeugt blockweise die Empfangsdaten aller sieben Antennenelemente fuer
den *Direktpfad* Sender -> Empfaenger. Bewusst noch nicht enthalten:
Rauschen, Zielecho, Doppler, Laufzeit.

Kette:
    FM-Basisband s(t)          (transmitter.FMStream, blockweise)
      x  Steering-Vektor a     (array.steering_vector, aus der Geometrie)
      -> Kanalmatrix (7, block_size)

Aufruf:
    python uca_stream.py
"""

import numpy as np

from uca import (N_ELEMENTS, azimuth, element_phases, radius_from_spacing,
                 steering_vector, wavelength)
from receiver import receive_array
from transmitter import FMStream

# --- Parameter --------------------------------------------------------
FC_HZ = 100e6           # Traegerfrequenz des UKW-Senders [Hz]
FS_HZ = 240e3           # Abtastrate des Basisbands [Hz]
N = N_ELEMENTS          # Anzahl Antennenelemente (7)
D_OVER_LAMBDA = 0.4     # Elementabstand d als Vielfaches von lambda
ARRAY_RADIUS_M = None   # r direkt vorgeben; None -> aus d berechnen
TX_POS_M = (5_000.0, 3_000.0)   # Senderposition [m]
RX_POS_M = (0.0, 0.0)           # Arrayzentrum des Empfaengers [m]
DEVIATION_HZ = 75e3     # Frequenzhub der FM [Hz]
BLOCK_SIZE = 4096       # Samples je Streaming-Block
N_BLOCKS = 20           # Anzahl Bloecke in der Demo


def fm_uca_stream(fc=FC_HZ, fs=FS_HZ, n=N, d_over_lambda=D_OVER_LAMBDA,
                  r=ARRAY_RADIUS_M, tx_pos=TX_POS_M, rx_pos=RX_POS_M,
                  deviation_hz=DEVIATION_HZ, block_size=BLOCK_SIZE,
                  n_blocks=None, mode="tones"):
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
    """Geometriegroessen des Streams (lambda, r, phi_tx, alpha, a)."""
    lam = wavelength(fc)
    if r is None:
        r = radius_from_spacing(d_over_lambda * lam, n)
    phi_tx = azimuth(tx_pos, rx_pos)
    alpha = element_phases(phi_tx, fc, r, n)
    return dict(lam=lam, r=r, phi_tx=phi_tx, alpha=alpha,
                a=np.exp(1j * alpha))


if __name__ == "__main__":
    g = stream_info()

    print("=== Geometrie ===")
    print(f"  fc            = {FC_HZ/1e6:.1f} MHz   ->  lambda = {g['lam']:.3f} m")
    print(f"  Elementabstand d = {D_OVER_LAMBDA} * lambda "
          f"= {D_OVER_LAMBDA*g['lam']:.3f} m")
    print(f"  Arrayradius   r = {g['r']:.3f} m  (N = {N} Elemente)")
    print(f"  Sender bei    {TX_POS_M} m,  Empfaenger bei {RX_POS_M} m")
    print(f"  Azimut phi_tx = {np.degrees(g['phi_tx']):.3f} deg")
    print("  alpha_n [deg] = "
          + ", ".join(f"{np.degrees(x):+.2f}" for x in g['alpha']))

    # Sollwert fuer die Kontrolle: Phasendifferenz Kanal 1 gegen Kanal 0
    soll = np.angle(np.exp(1j * (g['alpha'][1] - g['alpha'][0])))
    print(f"\n  erwartete Phasendifferenz Kanal 1 - Kanal 0 = "
          f"{np.degrees(soll):+.4f} deg")

    print(f"\n=== Datenstrom ({N_BLOCKS} Bloecke a {BLOCK_SIZE} Samples) ===")
    max_abw = 0.0
    for i, block in enumerate(fm_uca_stream(n_blocks=N_BLOCKS)):
        # Phasendifferenz zwischen Kanal 0 und 1: Kanal 1 * conj(Kanal 0)
        # eliminiert das gemeinsame Signal s(t); uebrig bleibt a_1*conj(a_0),
        # also eine ueber die ganze Zeit *konstante* Phase.
        dphi = np.angle(block[1] * np.conj(block[0]))
        abw = np.max(np.abs(np.angle(np.exp(1j * (dphi - soll)))))
        max_abw = max(max_abw, abw)
        print(f"  Block {i:2d}: shape={block.shape}  dtype={block.dtype}  "
              f"dphi(1-0) = {np.degrees(dphi.mean()):+.4f} deg  "
              f"(Streuung {np.degrees(dphi.std()):.2e} deg)")

    print(f"\n  max. Abweichung vom Sollwert ueber alle Bloecke: "
          f"{np.degrees(max_abw):.2e} deg")

    # Kontrolle der Blockgrenzen: bei fortgefuehrter Phase darf der Sprung
    # zwischen letztem und erstem Sample benachbarter Bloecke nicht groesser
    # sein als der typische Sprung *innerhalb* eines Blocks.
    blocks = list(fm_uca_stream(n_blocks=3))
    innen = np.abs(np.angle(blocks[0][0, 1:] * np.conj(blocks[0][0, :-1]))).max()
    grenze = np.abs(np.angle(blocks[1][0, 0] * np.conj(blocks[0][0, -1])))
    print(f"  Phasensprung an der Blockgrenze: {np.degrees(grenze):.3f} deg  "
          f"(max. innerhalb eines Blocks: {np.degrees(innen):.3f} deg)")

    # Betrag: reine FM -> konstante Einhuellende auf jedem Kanal
    print(f"  |X| min/max = {np.abs(blocks[0]).min():.4f} / "
          f"{np.abs(blocks[0]).max():.4f}  (soll ~1)")
