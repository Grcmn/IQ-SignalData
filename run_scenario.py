"""
Beispielszenario für den Passivradar-Signalgenerator (prsim).

Erzeugt die I/Q-Daten von Referenz- und Überwachungskanal für ein
DVB-T-Passivradar mit drei Flugzeugen und statischem Clutter,
speichert sie als .npy + Ground-Truth-JSON und validiert das Ergebnis
über eine Range-Doppler-Map (Batch-CAF).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prsim import (Transmitter, Receiver, Target, StaticScatterer,
                   Scenario, generate, batch_caf, C0)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ── 1. Szenario definieren ─────────────────────────────────────────────
# Koordinaten: lokales ENU-System in Metern, z = Höhe.

tx = Transmitter(pos=np.array([0.0, 0.0, 250.0]),      # DVB-T-Mast
                 erp_w=50e3, freq_hz=626e6)             # Kanal 40, 50 kW ERP

rx = Receiver(pos=np.array([25e3, 0.0, 30.0]),          # 25 km Basislinie
              gain_surv_db=8.0, gain_ref_db=12.0,
              noise_figure_db=6.0, dpi_suppression_db=40.0)

targets = [
    Target("A320 anfliegend", pos0=np.array([12e3, 15e3, 9000.0]),
           vel=np.array([-30.0, -220.0, 0.0]), rcs_m2=40.0),
    Target("B737 abfliegend", pos0=np.array([8e3, -6e3, 7500.0]),
           vel=np.array([150.0, -160.0, 3.0]), rcs_m2=30.0),
    Target("C172 langsam", pos0=np.array([18e3, 8e3, 1200.0]),
           vel=np.array([-45.0, -35.0, 0.0]), rcs_m2=2.0),
]

scatterers = [
    StaticScatterer("Industriegebiet", np.array([5e3, 2e3, 60.0]), 5000.0),
    StaticScatterer("Windpark", np.array([20e3, -3e3, 80.0]), 8000.0),
    StaticScatterer("Bergrücken", np.array([12e3, 10e3, 100.0]), 3000.0),
]

sc = Scenario(tx=tx, rx=rx, targets=targets, scatterers=scatterers,
              n_samples=2 ** 21, seed=7,
              adc_bits=12, lo_linewidth_hz=50.0, shared_lo=True)

# ── 2. I/Q-Daten erzeugen und speichern ────────────────────────────────
result = generate(sc)
ref, surv, fs = result["ref"], result["surv"], result["fs"]

np.save(OUT / "ref_channel.npy", ref.astype(np.complex64))
np.save(OUT / "surv_channel.npy", surv.astype(np.complex64))
with open(OUT / "scenario_groundtruth.json", "w") as f:
    json.dump({"fs_hz": fs, "fc_hz": result["fc"], "cpi_s": result["cpi"],
               "targets": result["ground_truth"]}, f, indent=2)
print("\nGespeichert: output/ref_channel.npy, output/surv_channel.npy, "
      "output/scenario_groundtruth.json")


# ── 3. Validierung: Range-Doppler-Map über Batch-CAF ───────────────────
print("\nBerechne Range-Doppler-Map zur Validierung ...")
n_delay = 800
caf, doppler_axis = batch_caf(ref, surv, fs, n_delay=n_delay, batch_len=4096)
caf_db = 20 * np.log10(np.abs(caf) + 1e-12)
caf_db -= caf_db.max()

range_axis_km = np.arange(n_delay) * C0 / fs / 1e3   # bistatische Range

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                               gridspec_kw={"height_ratios": [1, 2.2]})

# Spektrum des Referenzkanals (DVB-T: 7,61 MHz nutzbare Bandbreite)
f_axis = np.fft.fftshift(np.fft.fftfreq(2 ** 16, 1 / fs)) / 1e6
psd = np.abs(np.fft.fftshift(np.fft.fft(ref[:2 ** 16]))) ** 2
psd_db = 10 * np.log10(psd / psd.max() + 1e-12)
ax1.plot(f_axis, psd_db, lw=0.4)
ax1.set(title="Spektrum Referenzkanal (DVB-T, 8-MHz-Kanal)",
        xlabel="Frequenz [MHz]", ylabel="rel. Leistung [dB]",
        ylim=(-60, 3))
ax1.grid(True, alpha=0.4)

im = ax2.imshow(caf_db, aspect="auto", origin="lower", cmap="jet",
                extent=[range_axis_km[0], range_axis_km[-1],
                        doppler_axis[0], doppler_axis[-1]],
                vmin=-75, vmax=0)
fig.colorbar(im, ax=ax2, label="rel. Amplitude [dB]")
ax2.set(title="Range-Doppler-Map (Batch-CAF, unbereinigt: DPI/Clutter "
              "bei 0 Hz)",
        xlabel="Bistatische Range [km]", ylabel="Doppler [Hz]",
        ylim=(-700, 700))

for gt in result["ground_truth"]:
    r_km = gt["bistatic_range_m"] / 1e3
    ax2.plot(r_km, gt["doppler_hz"], "wo", mfc="none", ms=16, mew=1.8)
    ax2.annotate(gt["name"], (r_km, gt["doppler_hz"]),
                 xytext=(10, 12), textcoords="offset points",
                 color="white", weight="bold", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "rd_map_validation.png", dpi=130)
print("Plot gespeichert: output/rd_map_validation.png")

# Peak-Kontrolle: liegt in der Nähe jeder Ground-Truth-Position ein Maximum?
print("\nPeak-Verifikation (Suchfenster ±5 Range-Bins / ±4 Doppler-Bins):")
noise_floor = np.median(caf_db)
for gt in result["ground_truth"]:
    ri = int(round(gt["delay_samples"]))
    di = int(np.argmin(np.abs(doppler_axis - gt["doppler_hz"])))
    win = caf_db[max(di - 4, 0):di + 5, max(ri - 5, 0):ri + 6]
    print(f"  {gt['name']:<18} Peak {win.max():6.1f} dB "
          f"({win.max() - noise_floor:4.1f} dB über Rauschteppich)")
