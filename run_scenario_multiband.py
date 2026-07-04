"""
Multiband-Szenario: dieselbe Flugzeugszene, beobachtet über die drei
Empfangsantennen des Passivradars (FM, DAB, DVB-T) mit jeweils eigenem
Illuminator (drei verschiedene Sendemasten).

Erzeugt pro Band die I/Q-Daten (ref/surv als .npy + Ground-Truth-JSON)
und eine Vergleichsgrafik der drei Range-Doppler-Maps — sie zeigt
unmittelbar die unterschiedliche Range-Auflösung (~1 km / ~150 m / ~40 m)
und die signalspezifischen Artefakte der drei Illuminatoren.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prsim import (Transmitter, Receiver, Target, StaticScatterer, Scenario,
                   generate, batch_caf, C0, FS_DVBT, FS_DAB)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ── Gemeinsame Szene ───────────────────────────────────────────────────
rx = Receiver(pos=np.array([25e3, 0.0, 30.0]),
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
]

# ── Drei Illuminatoren mit typischen Parametern ────────────────────────
# (fs, n_samples, batch_len, n_delay so gewählt, dass Doppler eindeutig
#  bleibt und die relevanten bistatischen Ranges abgedeckt sind)
bands = [
    dict(name="FM 98,5 MHz",
         sc=Scenario(illuminator="fm", fs=300e3, n_samples=2 ** 18,  # 0,87 s
                     fm_content="music",
                     tx=Transmitter(np.array([-5e3, 8e3, 300.0]),
                                    erp_w=100e3, freq_hz=98.5e6),
                     rx=rx, targets=targets, scatterers=scatterers, seed=7),
         batch_len=1024, n_delay=40),
    dict(name="DAB 227,36 MHz (Block 12C)",
         sc=Scenario(illuminator="dab", fs=FS_DAB, n_samples=2 ** 19,  # 0,26 s
                     tx=Transmitter(np.array([3e3, -10e3, 250.0]),
                                    erp_w=10e3, freq_hz=227.36e6),
                     rx=rx, targets=targets, scatterers=scatterers, seed=7),
         batch_len=2048, n_delay=250),
    dict(name="DVB-T 626 MHz (Kanal 40)",
         sc=Scenario(illuminator="dvbt", fs=FS_DVBT, n_samples=2 ** 21,  # 0,23 s
                     tx=Transmitter(np.array([0.0, 0.0, 250.0]),
                                    erp_w=50e3, freq_hz=626e6),
                     rx=rx, targets=targets, scatterers=scatterers, seed=7),
         batch_len=4096, n_delay=800),
]

# ── Generieren, speichern, validieren ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
meta = {}

for band, ax in zip(bands, axes):
    sc = band["sc"]
    tag = sc.illuminator
    print(f"\n{'=' * 60}\n{band['name']}\n{'=' * 60}")
    res = generate(sc)

    np.save(OUT / f"ref_{tag}.npy", res["ref"].astype(np.complex64))
    np.save(OUT / f"surv_{tag}.npy", res["surv"].astype(np.complex64))
    meta[tag] = {"band": band["name"], "fs_hz": res["fs"],
                 "fc_hz": res["fc"], "cpi_s": res["cpi"],
                 "targets": res["ground_truth"]}

    caf, dopp = batch_caf(res["ref"], res["surv"], res["fs"],
                          n_delay=band["n_delay"], batch_len=band["batch_len"])
    caf_db = 20 * np.log10(np.abs(caf) + 1e-12)
    caf_db -= caf_db.max()
    rng_km = np.arange(band["n_delay"]) * C0 / res["fs"] / 1e3

    im = ax.imshow(caf_db, aspect="auto", origin="lower", cmap="jet",
                   extent=[rng_km[0], rng_km[-1], dopp[0], dopp[-1]],
                   vmin=-70, vmax=0)
    ax.set(title=f"{band['name']}\nRange-Bin: {C0/res['fs']:.0f} m, "
                 f"CPI: {res['cpi']*1e3:.0f} ms",
           xlabel="Bistatische Range [km]", ylabel="Doppler [Hz]")

    noise_floor = np.median(caf_db)
    for gt in res["ground_truth"]:
        r_km = gt["bistatic_range_m"] / 1e3
        ax.plot(r_km, gt["doppler_hz"], "wo", mfc="none", ms=14, mew=1.6)
        ri = int(round(gt["delay_samples"]))
        di = int(np.argmin(np.abs(dopp - gt["doppler_hz"])))
        win = caf_db[max(di - 4, 0):di + 5,
                     max(ri - 5, 0):min(ri + 6, band["n_delay"])]
        print(f"  Peak {gt['name']:<18} {win.max() - noise_floor:5.1f} dB "
              f"über Rauschteppich")

fig.colorbar(im, ax=axes, label="rel. Amplitude [dB]", fraction=0.02)
fig.suptitle("Dieselbe Szene über drei Illuminatoren — "
             "Range-Auflösung skaliert mit der Signalbandbreite", y=1.0)
plt.savefig(OUT / "rd_map_multiband.png", dpi=130, bbox_inches="tight")
print("\nPlot gespeichert: output/rd_map_multiband.png")

with open(OUT / "scenario_groundtruth_multiband.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Gespeichert: output/ref/surv_{fm,dab,dvbt}.npy, "
      "output/scenario_groundtruth_multiband.json")
