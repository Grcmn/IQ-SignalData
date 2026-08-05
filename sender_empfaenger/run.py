"""
Einstiegspunkt: IQ-Datenstrom am 7-Element-UCA erzeugen und pruefen.

Ablauf:
    1. Geometrie des Arrays und Einfallsrichtung des Senders ausgeben
    2. Datenstrom blockweise erzeugen (stream.fm_uca_stream)
    3. Kontrollen: Phasendifferenzen, Blockgrenzen, Einhuellende,
       Rueckgewinnung des modulierenden Signals m(t)
    4. optional Kontrollgrafiken (plots.py)

Aufruf:
    python run.py
    python run.py --plot
"""

import argparse

import numpy as np

from receiver import fm_demodulate
from stream import (BLOCK_SIZE, DEVIATION_HZ, FC_HZ, FS_HZ, MODE, N, N_BLOCKS,
                    D_OVER_LAMBDA, RX_POS_M, TX_POS_M, fm_uca_stream,
                    stream_info)
from transmitter import FMStream


def main(plot=False):
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

    # --- Datenstrom ---------------------------------------------------
    print(f"\n=== Datenstrom ({N_BLOCKS} Bloecke a {BLOCK_SIZE} Samples) ===")
    max_abw = 0.0
    blocks = []
    for i, block in enumerate(fm_uca_stream(n_blocks=N_BLOCKS, mode=MODE)):
        blocks.append(block)
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
    innen = np.abs(np.angle(blocks[0][0, 1:] * np.conj(blocks[0][0, :-1]))).max()
    grenze = np.abs(np.angle(blocks[1][0, 0] * np.conj(blocks[0][0, -1])))
    print(f"  Phasensprung an der Blockgrenze: {np.degrees(grenze):.3f} deg  "
          f"(max. innerhalb eines Blocks: {np.degrees(innen):.3f} deg)")

    # Betrag: reine FM -> konstante Einhuellende auf jedem Kanal
    print(f"  |X| min/max = {np.abs(blocks[0]).min():.4f} / "
          f"{np.abs(blocks[0]).max():.4f}  (soll ~1)")

    # --- Demodulations-Kontrolle --------------------------------------
    # Der gesamte Strom als eine Matrix (N, N_BLOCKS*BLOCK_SIZE)
    x = np.concatenate(blocks, axis=1)

    # Referenz: dasselbe m(t) noch einmal auf derselben Zeitachse erzeugen
    ref = FMStream(FS_HZ, deviation_hz=DEVIATION_HZ, mode=MODE)
    t = np.arange(x.shape[1]) / FS_HZ
    m_soll = ref._modulation(t)

    # fm_demodulate bildet Differenzen -> ein Sample kuerzer
    m_hat = fm_demodulate(x[0], FS_HZ, DEVIATION_HZ)
    fehler = m_hat - m_soll[1:]
    rms = np.sqrt(np.mean(fehler**2)) / np.sqrt(np.mean(m_soll**2))
    print("\n=== Demodulations-Kontrolle (Kanal 0) ===")
    print(f"  m(t) rueckgewonnen: relativer RMS-Fehler = {rms*100:.4f} %")

    # --- Grafik -------------------------------------------------------
    if plot:
        import plots
        print("\n=== Grafiken ===")
        print("  gespeichert:", plots.plot_signal(x, FS_HZ, mode=MODE,
                                                  deviation_hz=DEVIATION_HZ))
        print("  gespeichert:", plots.plot_array(x, g))

    return x


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true",
                   help="Kontrollgrafiken erzeugen (signal.png, array.png)")
    main(plot=p.parse_args().plot)
