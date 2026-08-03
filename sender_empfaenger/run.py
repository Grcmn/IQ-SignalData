"""
Demo Schritt 1: FM-Sender -> idealer Empfaenger -> Kontroll-Demodulation.

Ablauf:
    1. Sendesignal erzeugen (transmitter.generate_fm)
    2. Ideal empfangen             (receiver.receive)
    3. Zur Validierung demodulieren (receiver.fm_demodulate)
    4. Ergebnisse anschauen: Zeitsignal, Spektren, Rueckgewinnung

Aufruf:
    python run.py
"""

import numpy as np
import matplotlib.pyplot as plt

from transmitter import generate_fm, PILOT_HZ, SUBCARRIER_HZ
from receiver import receive, fm_demodulate

# --- Parameter --------------------------------------------------------
FS = 400e3           # Abtastrate 400 kHz (> Carson-Bandbreite ~256 kHz)
DAUER_S = 0.02       # Signaldauer in Sekunden
SEED = 1

n_samples = int(FS * DAUER_S)

# --- 1) Senden --------------------------------------------------------
tx_iq, tx_mpx = generate_fm(n_samples, FS, seed=SEED, return_mpx=True)
print(f"Gesendet:  {n_samples} I/Q-Samples bei fs = {FS/1e3:.0f} kHz")
print(f"           mittlere Leistung = {np.mean(np.abs(tx_iq)**2):.4f}  "
      f"(soll ~1)")
print(f"           |iq| konstant?    min={np.abs(tx_iq).min():.4f}  "
      f"max={np.abs(tx_iq).max():.4f}  (soll ~1, konstante Einhuellende)")

# --- 2) Empfangen (ideal) --------------------------------------------
rx_iq = receive(tx_iq)
print(f"Empfangen: identisch zum Sendesignal? {np.allclose(rx_iq, tx_iq)}")

# --- 3) Demodulieren (Kontrolle) -------------------------------------
rx_mpx = fm_demodulate(rx_iq, FS)
# Fehler zwischen gesendetem und zurueckgewonnenem MPX
fehler = rx_mpx - tx_mpx[1:]
rms = np.sqrt(np.mean(fehler**2)) / np.sqrt(np.mean(tx_mpx**2))
print(f"MPX-Rueckgewinnung: relativer RMS-Fehler = {rms*100:.3f} %")

# --- 4) Darstellung ---------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# (a) I/Q im Zeitbereich (kurzer Ausschnitt)
n_show = 200
ax[0, 0].plot(np.arange(n_show) / FS * 1e6, tx_iq[:n_show].real, label="I")
ax[0, 0].plot(np.arange(n_show) / FS * 1e6, tx_iq[:n_show].imag, label="Q")
ax[0, 0].set_title("Sendesignal I/Q (Zeitbereich)")
ax[0, 0].set_xlabel("Zeit [us]")
ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].legend()

# (b) Betrag der Einhuellenden (soll konstant = 1 sein)
ax[0, 1].plot(np.arange(n_show) / FS * 1e6, np.abs(tx_iq[:n_show]))
ax[0, 1].set_title("Einhuellende |iq| (konstant -> FM)")
ax[0, 1].set_xlabel("Zeit [us]")
ax[0, 1].set_ylabel("|iq|")
ax[0, 1].set_ylim(0, 1.5)

# (c) Leistungsspektrum des FM-Signals
spec = np.fft.fftshift(np.fft.fft(tx_iq * np.hanning(n_samples)))
f = np.fft.fftshift(np.fft.fftfreq(n_samples, 1 / FS)) / 1e3
psd = 20 * np.log10(np.abs(spec) + 1e-12)
ax[1, 0].plot(f, psd - psd.max())
ax[1, 0].set_title("Spektrum des FM-Sendesignals")
ax[1, 0].set_xlabel("Frequenz [kHz]")
ax[1, 0].set_ylabel("Leistung [dB]")
ax[1, 0].set_ylim(-80, 5)

# (d) Spektrum des demodulierten MPX: Pilot bei 19 kHz + Stereo um 38 kHz
mpx_spec = np.fft.rfft(rx_mpx * np.hanning(len(rx_mpx)))
mpx_f = np.fft.rfftfreq(len(rx_mpx), 1 / FS) / 1e3
mpx_psd = 20 * np.log10(np.abs(mpx_spec) + 1e-12)
ax[1, 1].plot(mpx_f, mpx_psd - mpx_psd.max())
ax[1, 1].axvline(PILOT_HZ / 1e3, color="r", ls="--", lw=1, label="19 kHz Pilot")
ax[1, 1].axvline(SUBCARRIER_HZ / 1e3, color="g", ls="--", lw=1,
                 label="38 kHz Stereo")
ax[1, 1].set_title("Demoduliertes MPX (Empfaenger)")
ax[1, 1].set_xlabel("Frequenz [kHz]")
ax[1, 1].set_ylabel("Leistung [dB]")
ax[1, 1].set_xlim(0, 60)
ax[1, 1].set_ylim(-80, 5)
ax[1, 1].legend()

fig.tight_layout()
fig.savefig("sender_empfaenger.png", dpi=120)
print("Grafik gespeichert: sender_empfaenger.png")
