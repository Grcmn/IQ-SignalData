import numpy as np
import matplotlib.pyplot as plt
import time
from Test3 import generate_passive_radar_scenario

def compute_fast_CAF(ref, surveillance, num_range_bins, num_doppler_bins, max_doppler, fs):
    """
    Schnelle Berechnung der CAF mittels FFT (Zirkulare Kreuzkorrelation),
    was für große Signalvektoren deutlich schneller ist als numpy.correlate().
    """
    N = len(ref)
    doppler_freqs = np.linspace(-max_doppler, max_doppler, num_doppler_bins)
    CAF = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)

    # FFT des Surveillance-Signals nur einmal berechnen
    surv_fft = np.fft.fft(surveillance)
    t_vec = np.arange(N) / fs

    print("Berechne CAF über Doppler-Bins...")
    start_time = time.time()
    for d_idx, f_d in enumerate(doppler_freqs):
        # Referenzsignal mit Doppler-Hypothese verschieben
        ref_doppler = ref * np.exp(1j * 2 * np.pi * f_d * t_vec)

        # Schnelle Kreuzkorrelation per FFT
        ref_fft = np.fft.fft(ref_doppler)
        corr = np.fft.ifft(surv_fft * np.conj(ref_fft))

        # Wir sind nur an den ersten 'num_range_bins' (positiven Delays) interessiert
        CAF[d_idx, :] = corr[:num_range_bins]

    print(f"CAF Berechnung abgeschlossen in {time.time() - start_time:.2f} Sekunden.")
    return CAF, doppler_freqs

if __name__ == "__main__":
    # 1. Szenario generieren (wir reduzieren CPI auf 0.1 Sekunden, damit es schneller läuft)
    print("Generiere Szenario-Signale...")
    fs = 2e6
    ref, surv, t = generate_passive_radar_scenario(fs=fs, CPI=0.1, SNR_dB=15)

    # 2. Analyse: CAF berechnen
    # Wir wählen maximalen Doppler 400Hz (deckt v=220m/s der B737 bei ~347 Hz ab)
    # Range Bins bis 300 (deckt das Delay von 120µs = 240 Samples des "Kleinflugzeugs" ab)
    CAF, doppler_axis = compute_fast_CAF(
        ref, surv,
        num_range_bins=300,
        num_doppler_bins=200,
        max_doppler=400,
        fs=fs
    )

    # 3. Pegel in dB umrechnen und normieren
    CAF_dB = 20 * np.log10(np.abs(CAF) + 1e-10)
    CAF_dB -= np.max(CAF_dB) # Maximalwert auf 0 dB setzen

    # 4. Range-Doppler-Map visualisieren
    plt.figure(figsize=(12, 7))
    range_axis = np.arange(300)

    plt.imshow(
        CAF_dB,
        aspect='auto',
        origin='lower',
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        cmap='jet',
        vmin=-60, # Begrenze auf -60dB Dynamik, so sieht man Rauschen dunkelblau
        vmax=0
    )
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Verzögerung / Range (Samples)')
    plt.ylabel('Doppler Frequenz [Hz]')
    plt.title('Passiv-Radar Range-Doppler-Map (erwartet: Clutter bei 0Hz, 3 Flugzeuge)')

    # Markierungen für die erwarteten Flugzeuge laut Test3.py einzeichnen
    targets = [
        {"name": "Boeing 737", "d_samp": 80, "f_d": 347.6},
        {"name": "A320", "d_samp": 160, "f_d": -237.0},
        {"name": "Kleinflugzeug", "d_samp": 240, "f_d": 126.4}
    ]
    for tgt in targets:
        plt.plot(tgt["d_samp"], tgt["f_d"], 'ko', markerfacecolor='none', markersize=15, markeredgewidth=2)
        plt.text(tgt["d_samp"] + 5, tgt["f_d"] + 15, tgt["name"], color='white', weight='bold', bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig('Range_Doppler_Map_Resultat.png')
    print("Ergebnis gespeichert als 'Range_Doppler_Map_Resultat.png'")
    # plt.show() # Wird für den Agenten nicht blockieren

