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

from uca import (N_ELEMENTS, azimuth, element_phases, element_positions,
                 radius_from_spacing, steering_vector, wavelength)
from receiver import fm_demodulate, receive_array
from transmitter import DEFAULT_TONES, FMStream, PILOT_HZ, SUBCARRIER_HZ

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
MODE = "tones"          # Modulationsquelle: "tones" oder "mpx"


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


def plot_stream(x, fs, geo, mode="tones", fname="uca_stream.png"):
    """Kontrollgrafik zum erzeugten Datenstrom.

    Parameter
    ---------
    x : (n, M) complex   aneinandergehaengte Stream-Bloecke
    fs : float           Abtastrate [Hz]
    geo : dict           Rueckgabe von stream_info()
    mode : str           Modulationsquelle (bestimmt die Achse in Panel b)

    matplotlib wird bewusst erst hier importiert, damit der Generator
    fm_uca_stream() ohne Plot-Abhaengigkeit importierbar bleibt.
    """
    import matplotlib.pyplot as plt

    n, M = x.shape
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Spektrum des empfangenen FM-Signals auf Kanal 0.
    #     Hinweis: bei fs = 240 kHz liegt die Carson-Bandbreite (~256 kHz)
    #     knapp ausserhalb des abgetasteten Bandes — das Spektrum stoesst
    #     sichtbar an die Bandgrenzen. Fuer die Geometriepruefung ohne
    #     Belang, fuer spaetere Kreuzkorrelation lieber fs = 400 kHz.
    spec = np.fft.fftshift(np.fft.fft(x[0] * np.hanning(M)))
    f = np.fft.fftshift(np.fft.fftfreq(M, 1 / fs)) / 1e3
    psd = 20 * np.log10(np.abs(spec) + 1e-12)
    ax[0, 0].plot(f, psd - psd.max(), lw=0.8)
    carson = (2 * (DEVIATION_HZ + 15e3)) / 2 / 1e3      # halbe Carson-BW [kHz]
    for s in (-carson, carson):
        ax[0, 0].axvline(s, color="r", ls="--", lw=1)
    ax[0, 0].set_title("(a) Empfangenes FM-Spektrum, Kanal 0\n"
                       "(rot: Carson-Bandbreite)")
    ax[0, 0].set_xlabel("Frequenz [kHz]")
    ax[0, 0].set_ylabel("Leistung [dB]")
    ax[0, 0].set_ylim(-80, 5)

    # (b) Spektrum des demodulierten Signals: zurueckgewonnenes m(t).
    #     Die konstante Steering-Phase a_0 verschwindet bei der Differenz-
    #     bildung der FM-Demodulation, m(t) kommt unveraendert heraus.
    m_hat = fm_demodulate(x[0], fs)
    m_spec = np.fft.rfft(m_hat * np.hanning(len(m_hat)))
    m_f = np.fft.rfftfreq(len(m_hat), 1 / fs) / 1e3
    m_psd = 20 * np.log10(np.abs(m_spec) + 1e-12)
    ax[0, 1].plot(m_f, m_psd - m_psd.max(), lw=0.8)
    # Marker hinter die Kurve legen (zorder=0), sonst verdecken sie genau
    # die Linien, die sie markieren sollen.
    if mode == "mpx":
        ax[0, 1].axvline(PILOT_HZ / 1e3, color="r", ls="--", lw=1, zorder=0,
                         label="19 kHz Pilot")
        ax[0, 1].axvline(SUBCARRIER_HZ / 1e3, color="g", ls="--", lw=1,
                         zorder=0, label="38 kHz Stereo")
        ax[0, 1].set_xlim(0, 60)
    else:
        for tone_hz, _ in DEFAULT_TONES:
            ax[0, 1].axvline(tone_hz / 1e3, color="r", ls="--", lw=1, zorder=0)
        ax[0, 1].set_xlim(0, 15)
        ax[0, 1].plot([], [], color="r", ls="--", lw=1, label="Solltoene")
    ax[0, 1].set_title("(b) Demoduliertes Signal m(t) — Spektrum")
    ax[0, 1].set_xlabel("Frequenz [kHz]")
    ax[0, 1].set_ylabel("Leistung [dB]")
    ax[0, 1].set_ylim(-80, 5)
    ax[0, 1].legend()

    # (c) Arraygeometrie mit Einfallsrichtung.
    #     Der Sender liegt einige Kilometer entfernt, also weit ausserhalb
    #     des Bildes — gezeigt wird nur seine Richtung als Pfeil.
    pos = element_positions(geo["r"], n)
    ax[1, 0].scatter(pos[:, 0], pos[:, 1], s=60, zorder=3, label="Elemente")
    for i, (px, py) in enumerate(pos):
        ax[1, 0].annotate(str(i), (px, py), textcoords="offset points",
                          xytext=(6, 6))
    ax[1, 0].scatter([0], [0], marker="+", s=120, color="k",
                     label="Arrayzentrum")
    arrow = 1.4 * geo["r"]
    ax[1, 0].arrow(0, 0, arrow * np.cos(geo["phi_tx"]),
                   arrow * np.sin(geo["phi_tx"]), width=0.02,
                   color="r", length_includes_head=True, zorder=2,
                   label=f"Sender ({np.degrees(geo['phi_tx']):.1f} deg)")
    ax[1, 0].set_title(f"(c) UCA-Geometrie, N = {n}, r = {geo['r']:.2f} m")
    ax[1, 0].set_xlabel("x [m]")
    ax[1, 0].set_ylabel("y [m]")
    ax[1, 0].set_aspect("equal")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].margins(0.22)          # Platz fuer die Elementbeschriftungen
    ax[1, 0].legend(loc="lower right", fontsize=8)

    # (d) Phasendifferenz jedes Kanals gegen Kanal 0.
    #     x[k] * conj(x[0]) eliminiert das gemeinsame s(t); uebrig bleibt
    #     a_k * conj(a_0), also exakt alpha_k - alpha_0.
    gemessen = np.degrees([np.angle(np.mean(x[k] * np.conj(x[0])))
                           for k in range(n)])
    soll = np.degrees(np.angle(np.exp(1j * (geo["alpha"] - geo["alpha"][0]))))
    idx = np.arange(n)
    ax[1, 1].plot(idx, soll, "o", ms=12, mfc="none", label="Soll (Geometrie)")
    ax[1, 1].plot(idx, gemessen, "x", ms=9, label="gemessen (Stream)")
    ax[1, 1].set_title("(d) Phasendifferenz Kanal n gegen Kanal 0")
    ax[1, 1].set_xlabel("Elementindex n")
    ax[1, 1].set_ylabel("Phase [deg]")
    ax[1, 1].set_xticks(idx)
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    return fname


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

    # --- Demodulations-Kontrolle -------------------------------------
    # Der gesamte Strom als eine Matrix (N, N_BLOCKS*BLOCK_SIZE)
    x = np.concatenate(blocks, axis=1)

    # Referenz: dasselbe m(t) noch einmal auf derselben Zeitachse erzeugen
    ref = FMStream(FS_HZ, deviation_hz=DEVIATION_HZ, mode=MODE)
    t = np.arange(x.shape[1]) / FS_HZ
    m_soll = ref._modulation(t)

    # fm_demodulate bildet Differenzen -> ein Sample kuerzer
    m_hat = fm_demodulate(x[0], FS_HZ)
    fehler = m_hat - m_soll[1:]
    rms = np.sqrt(np.mean(fehler**2)) / np.sqrt(np.mean(m_soll**2))
    print(f"\n=== Demodulations-Kontrolle (Kanal 0) ===")
    print(f"  m(t) rueckgewonnen: relativer RMS-Fehler = {rms*100:.4f} %")

    # --- Grafik -------------------------------------------------------
    fname = plot_stream(x, FS_HZ, g, mode=MODE)
    print(f"\nGrafik gespeichert: {fname}")
