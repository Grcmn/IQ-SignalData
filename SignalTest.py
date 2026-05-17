import numpy as np
import matplotlib.pyplot as plt

# ─── Systemparameter ───────────────────────────────────────────
fs = 2e6          # Abtastrate [Hz]
f_carrier = 474e6 # DVB-T Träger [Hz] (Kanal 21)
c = 3e8           # Lichtgeschwindigkeit
lambda_ = c / f_carrier  # Wellenlänge ~0.63m

CPI = 0.5         # Coherent Processing Interval [s]
N = int(fs * CPI) # Anzahl Samples
t = np.arange(N) / fs  # Zeitvektor

# ─── Zielparameter ─────────────────────────────────────────────
v_target = 200.0        # Radialgeschwindigkeit [m/s]
range_delay = 50e-6     # Laufzeitverzögerung [s] → ~15km bistatic range
SNR_dB = 20             # Signal-Rausch-Abstand

# ─── Referenzsignal erzeugen (vereinfacht: OFDM-ähnliches Rauschen) ──
# In der Praxis: echtes DVB-T Signal oder aufgezeichnetes Signal
np.random.seed(42)
ref_signal = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

# ─── Doppler-Frequenz berechnen ────────────────────────────────
f_doppler = v_target / lambda_
print(f"Doppler-Frequenz: {f_doppler:.1f} Hz")
print(f"Wellenlänge:      {lambda_*100:.1f} cm")

# ─── Echo-Signal erzeugen ──────────────────────────────────────
delay_samples = int(range_delay * fs)

# Verschobenes Referenzsignal (Range-Delay)
echo = np.zeros(N, dtype=complex)
echo[delay_samples:] = ref_signal[:N - delay_samples]

# Doppler-Modulation aufmultiplizieren
echo *= np.exp(1j * 2 * np.pi * f_doppler * t)

# Amplitude
alpha = 10 ** (SNR_dB / 20)
echo *= alpha

# ─── AWGN Rauschen addieren ────────────────────────────────────
noise_power = 1.0
noise = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
rx_signal = echo + noise

print(f"\nI/Q Signal erzeugt: {N} Samples")
print(f"Realteil  (I): min={rx_signal.real.min():.3f}, max={rx_signal.real.max():.3f}")
print(f"Imaginärteil (Q): min={rx_signal.imag.min():.3f}, max={rx_signal.imag.max():.3f}")

# ─── Visualisierung ────────────────────────────────────────────
# Wir zeigen hier nur einen kleinen Ausschnitt (die ersten 1000 Samples),
# um die Wellenform gut sichtbar zu machen.
show_samples = 1000

plt.figure(figsize=(12, 8))

# 1. Plot: I- und Q-Signal im Zeitbereich
plt.subplot(2, 1, 1)
plt.plot(t[:show_samples] * 1e6, rx_signal.real[:show_samples], label='I (In-Phase)')
plt.plot(t[:show_samples] * 1e6, rx_signal.imag[:show_samples], label='Q (Quadratur)', alpha=0.7)
plt.title('I/Q-Signal im Zeitbereich (Ausschnitt)')
plt.xlabel('Zeit [µs]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# 2. Plot: I/Q-Ebene (Konstellationsdiagramm / Scatterplot)
plt.subplot(2, 1, 2)
plt.scatter(rx_signal.real[:show_samples], rx_signal.imag[:show_samples], c='blue', alpha=0.5, s=10)
plt.title('I/Q-Ebene (Scatterplot)')
plt.xlabel('I (Realteil)')
plt.ylabel('Q (Imaginärteil)')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()
