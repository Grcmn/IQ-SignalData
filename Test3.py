import numpy as np

c = 3e8  # Lichtgeschwindigkeit in m/s

def generate_passive_radar_scenario(fs=2e6, CPI=0.5, SNR_dB=15):
    """
    Erzeugt komplettes I/Q Szenario:
    - Referenzsignal
    - Mehrere Flugzeuge
    - Clutter (stationäre Reflexionen)
    - Rauschen
    """
    N = int(fs * CPI)
    t = np.arange(N) / fs
    f_c = 474e6
    lambda_ = c / f_c

    # Referenzsignal (Pseudozufällig als DVB-T Ersatz)
    ref = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    surveillance = np.zeros(N, dtype=complex)

    # ── Ziele definieren ──────────────────────────────────────
    targets = [
        {"v": 220,  "delay_s": 40e-6,  "snr": 20, "name": "Boeing 737"},
        {"v": -150, "delay_s": 80e-6,  "snr": 15, "name": "A320"},
        {"v": 80,   "delay_s": 120e-6, "snr": 10, "name": "Kleinflugzeug"},
    ]

    for tgt in targets:
        f_d = tgt["v"] / lambda_
        d_samp = int(tgt["delay_s"] * fs)
        amp = 10 ** (tgt["snr"] / 20)

        echo = np.zeros(N, dtype=complex)
        echo[d_samp:] = ref[:N - d_samp]
        echo *= amp * np.exp(1j * 2 * np.pi * f_d * t)
        surveillance += echo
        print(f"{tgt['name']}: f_D = {f_d:.1f} Hz, delay = {d_samp} samples")

    # ── Clutter (stationär → f_D = 0) ─────────────────────────
    for delay_s, amp in [(5e-6, 50), (15e-6, 30), (25e-6, 20)]:
        d_samp = int(delay_s * fs)
        clutter = np.zeros(N, dtype=complex)
        clutter[d_samp:] = ref[:N - d_samp]
        clutter *= amp  # Clutter viel stärker als Ziele!
        surveillance += clutter

    # ── AWGN ──────────────────────────────────────────────────
    noise = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    surveillance += noise

    return ref, surveillance, t

# Szenario generieren
ref, surv, t = generate_passive_radar_scenario()

# Als .npy speichern (für spätere Verarbeitung)
np.save("ref_signal.npy", ref)
np.save("surveillance_signal.npy", surv)
print("\nGespeichert: ref_signal.npy, surveillance_signal.npy")