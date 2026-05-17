import numpy as np
import matplotlib.pyplot as plt

def compute_CAF(ref, surveillance, num_range_bins, num_doppler_bins, fs):
    """
    Kreuzambiguitätsfunktion: erzeugt Range-Doppler-Map

    ref:              Referenzsignal (direkte Welle)
    surveillance:     Überwachungskanal (Echo)
    num_range_bins:   Anzahl Range-Bins (Laufzeit-Delays)
    num_doppler_bins: Anzahl Doppler-Bins (Frequenzen)
    """
    N = len(ref)

    # Doppler-Frequenzachse
    doppler_freqs = np.linspace(-fs / 2, fs / 2, num_doppler_bins)

    CAF = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)

    for d_idx, f_d in enumerate(doppler_freqs):
        # Doppler-kompensiertes Referenzsignal
        t_vec = np.arange(N) / fs
        ref_doppler = ref * np.exp(-1j * 2 * np.pi * f_d * t_vec)

        # Kreuzkorrelation (effizient via FFT)
        corr = np.correlate(surveillance, ref_doppler, mode='full')

        # Mittlere N Samples (zero-lag zentriert)
        center = len(corr) // 2
        CAF[d_idx, :] = corr[center: center + num_range_bins]

    return CAF, doppler_freqs


# ─── Beispiel Setup ────────────────────────────────────────────
fs = 1e6            # Abtastrate (1 MHz)
duration = 0.005    # 5 ms Signallänge
N = int(fs * duration)
c = 3e8             # Lichtgeschwindigkeit (m/s)

# 1. Referenzsignal (z.B. komplexes Rauschen, typisch für DVB-T/DAB)
np.random.seed(42)
ref_signal = np.random.randn(N) + 1j * np.random.randn(N)

# 2. Überwachungskanal erstellen (Echo + Rauschen)
target_delay_samples = 150   # Entspricht einer gewissen Entfernung
target_doppler = 1500.0      # Dopplerfrequenz in Hz

t_vec = np.arange(N) / fs
rx_signal = np.zeros(N, dtype=complex)

# Echo verschieben
if target_delay_samples < N:
    rx_signal[target_delay_samples:] = ref_signal[:-target_delay_samples]

# Doppler aufprägen
rx_signal *= np.exp(1j * 2 * np.pi * target_doppler * t_vec)

# Grundrauschen hinzufügen
rx_signal += 0.5 * (np.random.randn(N) + 1j * np.random.randn(N))


# ─── CAF berechnen ─────────────────────────────────────────────
num_range_bins = 512
num_doppler_bins = 256

CAF, doppler_axis = compute_CAF(
    ref_signal, rx_signal,
    num_range_bins, num_doppler_bins, fs
)

CAF_dB = 20 * np.log10(np.abs(CAF) + 1e-10)

# ─── Visualisierung ────────────────────────────────────────────
range_axis = np.arange(num_range_bins) / fs * c / 2  # in Metern

plt.figure(figsize=(12, 5))
plt.imshow(
    CAF_dB,
    aspect='auto',
    extent=[range_axis[0], range_axis[-1],
            doppler_axis[-1], doppler_axis[0]],
    cmap='jet'
)
plt.colorbar(label='Amplitude [dB]')
plt.xlabel('Bistatic Range [m]')
plt.ylabel('Doppler Frequenz [Hz]')
plt.title('Range-Doppler-Map (CAF)')
plt.tight_layout()
plt.show()