# IQ-SignalData

Synthetische I/Q-Signaldaten für Passivradar-Simulation.

## Ziel dieses Repos

Dieses Repository hilft dir bei der Vorbereitung auf ein Gespräch zur Bachelorarbeit (z. B. bei HENSOLDT), wenn du **synthetische Testdaten für Passivradar** erzeugen sollst – speziell im Bereich **Signalverarbeitung mit Real- und Imaginärteil (I/Q)**.

---

## 1) Was bedeutet „I/Q-Signal“?

Ein I/Q-Signal ist ein **komplexes Basisbandsignal**

\[
s(t) = I(t) + j\,Q(t)
\]

- **I(t)** = In-Phase-Komponente (Realteil)
- **Q(t)** = Quadratur-Komponente (Imaginärteil)
- Betrag und Phase:
  - \(|s(t)| = \sqrt{I^2 + Q^2}\)
  - \(\phi(t) = \mathrm{atan2}(Q, I)\)

Für Radar ist das ideal, weil sich damit **Doppler, Phasenverschiebungen, Verzögerungen** und **Mischprodukte** sauber modellieren lassen.

---

## 2) Relevanz für Passivradar

Passivradar sendet selbst nicht, sondern nutzt fremde Sender („Illuminators of Opportunity“), z. B. FM/DAB/DVB-T/5G.

Typische Kanäle:
1. **Referenzkanal**: direkt empfangenes Sendersignal
2. **Überwachungskanal**: Echo vom Ziel + Clutter + Rauschen

Vereinfacht:

\[
x_{surv}(t)=\sum_k \alpha_k\,x_{ref}(t-\tau_k)\,e^{j2\pi f_{D,k}t} + c(t)+n(t)
\]

- \(\alpha_k\): Zielamplitude / RCS-Effekt
- \(\tau_k\): Laufzeitverzögerung (Range)
- \(f_{D,k}\): Dopplerfrequenz (Relativgeschwindigkeit)
- \(c(t)\): Clutter/Mehrwege
- \(n(t)\): Rauschen

---

## 3) Wie bereitest du dich fachlich am besten vor?

### A) Theorie (Kernpunkte)
- Komplexe Signale, IQ-Mischer, Basisbanddarstellung
- Abtastung, Nyquist, Alias-Effekte
- Rauschmodelle (AWGN), SNR/SCNR
- Delay & Doppler-Modellierung
- Kreuzkorrelation / Ambiguitätsfunktion / Range-Doppler-Map
- Clutter-Unterdrückung (z. B. einfache Canceller)

### B) Praktikumsnah
- Kleine reproduzierbare Simulationen bauen
- Parameter sauber dokumentieren (fs, CPI, SNR, Zielgeschwindigkeit, Delay)
- Datensätze mit Ground Truth erzeugen (Ziel da/weg, Doppler bekannt)

### C) Im Gespräch überzeugend zeigen
- Du verstehst die physikalische Bedeutung von Delay und Doppler
- Du kannst synthetische Daten erzeugen, die algorithmisch testbar sind
- Du kannst Grenzen benennen (vereinfacht vs. realistisch)

---

## 4) Python-Beispiel: synthetische I/Q-Testdaten

```python
import numpy as np

def db_to_lin(db):
    return 10 ** (db / 10.0)

# Parameter
fs = 200_000            # Abtastfrequenz [Hz]
T = 0.2                 # Signaldauer [s]
N = int(fs * T)
t = np.arange(N) / fs

# Referenzsignal (komplexes Basisbandsignal)
# Hier als Summe von Trägern + leichter FM-artiger Phasenmodulation
f1, f2 = 12e3, 27e3
phi = 2*np.pi*(200*np.sin(2*np.pi*30*t)) / fs
x_ref = (0.8*np.exp(1j*(2*np.pi*f1*t + phi)) +
         0.5*np.exp(1j*(2*np.pi*f2*t)))

# Zielparameter
delay_s = 180e-6        # Verzögerung [s]
delay_n = int(round(delay_s * fs))
f_d = 120.0             # Doppler [Hz]
alpha = 0.2             # Zielamplitude

# Verzögertes Referenzsignal (einfaches Integer-Delay)
x_del = np.roll(x_ref, delay_n)

# Zielanteil mit Doppler
x_target = alpha * x_del * np.exp(1j * 2*np.pi*f_d*t)

# Clutter (starker quasi-statischer Anteil)
x_clutter = 0.6 * x_ref

# Rauschen auf gewünschten SNR-Pegel
snr_db = 5
sig_pow = np.mean(np.abs(x_target + x_clutter)**2)
noise_pow = sig_pow / db_to_lin(snr_db)
noise = np.sqrt(noise_pow/2) * (np.random.randn(N) + 1j*np.random.randn(N))

# Überwachungskanal
x_surv = x_target + x_clutter + noise

# Export als getrennte I/Q Arrays
I_ref, Q_ref = np.real(x_ref), np.imag(x_ref)
I_surv, Q_surv = np.real(x_surv), np.imag(x_surv)

print("Shapes:", I_ref.shape, Q_ref.shape, I_surv.shape, Q_surv.shape)
```

### Hinweis
- Für realistischere Delays statt `np.roll` besser **fraktionale Delays** (z. B. FIR-Interpolation) verwenden.
- Für Training/Validierung mehrere Szenarien erzeugen (verschiedene Delays, Doppler, SNR, Anzahl Ziele).

---

## 5) MATLAB-Beispiel: synthetische I/Q-Testdaten

```matlab
fs = 200e3;
T = 0.2;
N = round(fs*T);
t = (0:N-1)/fs;

% Referenzsignal
f1 = 12e3; f2 = 27e3;
phi = 2*pi*(200*sin(2*pi*30*t))/fs;
x_ref = 0.8*exp(1j*(2*pi*f1*t + phi)) + 0.5*exp(1j*2*pi*f2*t);

% Zielparameter
delay_s = 180e-6;
delay_n = round(delay_s * fs);
f_d = 120;
alpha = 0.2;

x_del = circshift(x_ref, delay_n);
x_target = alpha * x_del .* exp(1j*2*pi*f_d*t);

% Clutter + Noise
x_clutter = 0.6 * x_ref;
snr_db = 5;
sig_pow = mean(abs(x_target + x_clutter).^2);
noise_pow = sig_pow / (10^(snr_db/10));
noise = sqrt(noise_pow/2) * (randn(1,N) + 1j*randn(1,N));

x_surv = x_target + x_clutter + noise;

I_ref = real(x_ref); Q_ref = imag(x_ref);
I_surv = real(x_surv); Q_surv = imag(x_surv);

disp(size(I_ref));
```

---

## 6) Mindest-Checkliste für „gute“ synthetische Testdaten

- [ ] Parameterbereich definiert (Delay, Doppler, SNR, Ziele)
- [ ] Referenz- und Überwachungskanal vorhanden
- [ ] Ground Truth je Datensatz gespeichert
- [ ] Mehrere Schwierigkeitsgrade (leicht → realistisch)
- [ ] Reproduzierbarkeit (Random Seed, Versionsstand)

---

## 7) Typische Fragen im Gespräch (und worauf du antworten solltest)

1. **Wie modellierst du ein Ziel im Passivradar?**  
   → Als verzögertes + Doppler-verschobenes Referenzsignal mit komplexer Amplitude.

2. **Warum I/Q statt nur Realteil?**  
   → Vollständige Phaseninformation und saubere Darstellung positiver/negativer Frequenzen.

3. **Wie validierst du die Datenqualität?**  
   → Korrelationspeak bei erwarteter Verzögerung, Dopplerpeak bei bekannter Geschwindigkeit, SNR-Regressionstests.

4. **Was fehlt in einfachen Modellen?**  
   → Kanal-/Antennencharakteristik, nichtstationäres Clutter, Hardware-Offsets, Quantisierung, Synchronisationsfehler.

---

## 8) Nächster Schritt

Wenn du willst, kann daraus direkt ein kleines Datengenerierungs-Template aufgebaut werden (CSV/NPY/MAT-Export, Szenario-Loop, Ground-Truth-JSON), das du für deine Bachelorarbeit und spätere Algorithmustests nutzen kannst.
