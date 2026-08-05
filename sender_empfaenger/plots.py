"""
Kontrollgrafiken zum erzeugten IQ-Datenstrom.

Alle matplotlib-Abhaengigkeiten des Projekts stehen ausschliesslich hier.
stream.py, transmitter.py, receiver.py und uca.py bleiben dadurch ohne
Plot-Abhaengigkeit importierbar.

Zwei Grafiken, nach Fragestellung getrennt:
    plot_signal()  — stimmt das FM-Signal?  (Zeitverlauf, Spektrum, Demod)
    plot_array()   — stimmt die Geometrie?  (UCA-Layout, Phasendifferenzen)

Aufruf ueber run.py --plot.
"""

import matplotlib
matplotlib.use("Agg")           # kein Fenster noetig, nur PNG-Ausgabe
import matplotlib.pyplot as plt
import numpy as np

from receiver import fm_demodulate
from transmitter import DEFAULT_TONES, PILOT_HZ, SUBCARRIER_HZ
from uca import element_positions


def plot_signal(x, fs, mode="mpx", deviation_hz=75e3, fname="signal.png"):
    """Grafik zum Signal selbst, ausgewertet auf Kanal 0.

    Parameter
    ---------
    x : (n, M) complex   Kanalmatrix des Streams
    fs : float           Abtastrate [Hz]
    mode : str           Modulationsquelle (bestimmt die Marker in Panel d)
    deviation_hz : float Frequenzhub (fuer Carson-Bandbreite und Demod)
    """
    ch0 = x[0]
    M = ch0.size
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (a) I/Q im Zeitbereich (kurzer Ausschnitt)
    n_show = min(200, M)
    t_us = np.arange(n_show) / fs * 1e6
    ax[0, 0].plot(t_us, ch0[:n_show].real, label="I")
    ax[0, 0].plot(t_us, ch0[:n_show].imag, label="Q")
    ax[0, 0].set_title("(a) I/Q-Datenstrom, Kanal 0 (Zeitbereich)")
    ax[0, 0].set_xlabel("Zeit [us]")
    ax[0, 0].set_ylabel("Amplitude")
    ax[0, 0].legend()

    # (b) Betrag der Einhuellenden — bei reiner FM konstant 1
    ax[0, 1].plot(t_us, np.abs(ch0[:n_show]))
    ax[0, 1].set_title("(b) Einhuellende |x| (konstant -> reine FM)")
    ax[0, 1].set_xlabel("Zeit [us]")
    ax[0, 1].set_ylabel("|x|")
    ax[0, 1].set_ylim(0, 1.5)

    # (c) Spektrum des empfangenen FM-Signals.
    #     Hinweis: bei fs = 240 kHz liegt die Carson-Bandbreite (~256 kHz)
    #     knapp ausserhalb des abgetasteten Bandes — das Spektrum stoesst
    #     sichtbar an die Bandgrenzen. Fuer die Geometriepruefung ohne
    #     Belang, fuer spaetere Kreuzkorrelation lieber fs = 400 kHz.
    spec = np.fft.fftshift(np.fft.fft(ch0 * np.hanning(M)))
    f = np.fft.fftshift(np.fft.fftfreq(M, 1 / fs)) / 1e3
    psd = 20 * np.log10(np.abs(spec) + 1e-12)
    ax[1, 0].plot(f, psd - psd.max(), lw=0.8)
    carson = (deviation_hz + 15e3) / 1e3        # halbe Carson-BW [kHz]
    for s in (-carson, carson):
        ax[1, 0].axvline(s, color="r", ls="--", lw=1)
    ax[1, 0].set_title("(c) FM-Spektrum, Kanal 0\n(rot: Carson-Bandbreite)")
    ax[1, 0].set_xlabel("Frequenz [kHz]")
    ax[1, 0].set_ylabel("Leistung [dB]")
    ax[1, 0].set_ylim(-80, 5)

    # (d) Spektrum des demodulierten Signals: zurueckgewonnenes m(t).
    #     Die konstante Steering-Phase a_0 verschwindet bei der Differenz-
    #     bildung der FM-Demodulation, m(t) kommt unveraendert heraus.
    m_hat = fm_demodulate(ch0, fs, deviation_hz)
    m_spec = np.fft.rfft(m_hat * np.hanning(len(m_hat)))
    m_f = np.fft.rfftfreq(len(m_hat), 1 / fs) / 1e3
    m_psd = 20 * np.log10(np.abs(m_spec) + 1e-12)
    ax[1, 1].plot(m_f, m_psd - m_psd.max(), lw=0.8)
    # Marker hinter die Kurve legen (zorder=0), sonst verdecken sie genau
    # die Linien, die sie markieren sollen.
    if mode == "mpx":
        ax[1, 1].axvline(PILOT_HZ / 1e3, color="r", ls="--", lw=1, zorder=0,
                         label="19 kHz Pilot")
        ax[1, 1].axvline(SUBCARRIER_HZ / 1e3, color="g", ls="--", lw=1,
                         zorder=0, label="38 kHz Stereo")
        ax[1, 1].set_xlim(0, 60)
    else:
        for tone_hz, _ in DEFAULT_TONES:
            ax[1, 1].axvline(tone_hz / 1e3, color="r", ls="--", lw=1, zorder=0)
        ax[1, 1].set_xlim(0, 15)
        ax[1, 1].plot([], [], color="r", ls="--", lw=1, label="Solltoene")
    ax[1, 1].set_title("(d) Demoduliertes m(t) — Spektrum")
    ax[1, 1].set_xlabel("Frequenz [kHz]")
    ax[1, 1].set_ylabel("Leistung [dB]")
    ax[1, 1].set_ylim(-80, 5)
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


def plot_array(x, geo, fname="array.png"):
    """Grafik zur Arraygeometrie: Layout und gemessene Phasendifferenzen.

    Parameter
    ---------
    x : (n, M) complex   Kanalmatrix des Streams
    geo : dict           Rueckgabe von stream.stream_info()
    """
    n = x.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Arraygeometrie mit Einfallsrichtung.
    #     Der Sender liegt einige Kilometer entfernt, also weit ausserhalb
    #     des Bildes — gezeigt wird nur seine Richtung als Pfeil.
    pos = element_positions(geo["r"], n)
    ax[0].scatter(pos[:, 0], pos[:, 1], s=60, zorder=3, label="Elemente")
    for i, (px, py) in enumerate(pos):
        ax[0].annotate(str(i), (px, py), textcoords="offset points",
                       xytext=(6, 6))
    ax[0].scatter([0], [0], marker="+", s=120, color="k",
                  label="Arrayzentrum")
    arrow = 1.4 * geo["r"]
    ax[0].arrow(0, 0, arrow * np.cos(geo["phi_tx"]),
                arrow * np.sin(geo["phi_tx"]), width=0.02,
                color="r", length_includes_head=True, zorder=2,
                label=f"Sender ({np.degrees(geo['phi_tx']):.1f} deg)")
    ax[0].set_title(f"(a) UCA-Geometrie, N = {n}, r = {geo['r']:.2f} m")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[0].margins(0.22)          # Platz fuer die Elementbeschriftungen
    ax[0].legend(loc="lower right", fontsize=8)

    # (b) Phasendifferenz jedes Kanals gegen Kanal 0.
    #     x[k] * conj(x[0]) eliminiert das gemeinsame s(t); uebrig bleibt
    #     a_k * conj(a_0), also exakt alpha_k - alpha_0.
    gemessen = np.degrees([np.angle(np.mean(x[k] * np.conj(x[0])))
                           for k in range(n)])
    soll = np.degrees(np.angle(np.exp(1j * (geo["alpha"] - geo["alpha"][0]))))
    idx = np.arange(n)
    ax[1].plot(idx, soll, "o", ms=12, mfc="none", label="Soll (Geometrie)")
    ax[1].plot(idx, gemessen, "x", ms=9, label="gemessen (Stream)")
    ax[1].set_title("(b) Phasendifferenz Kanal n gegen Kanal 0")
    ax[1].set_xlabel("Elementindex n")
    ax[1].set_ylabel("Phase [deg]")
    ax[1].set_xticks(idx)
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname
