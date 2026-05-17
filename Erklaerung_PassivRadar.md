# Passiv-Radar und die Cross-Ambiguity Function (CAF)

Dieses Dokument erklärt die Funktionsweise des simulierten Passiv-Radar-Szenarios und der Auswertung mittels der *Cross-Ambiguity Function* (CAF) basierend auf den Python-Skripten `Test3.py` und `analyze_scenario.py`.

---

## 1. Das Konzept

Ein Passiv-Radar nutzt sogenannte Fremdstrahler (z. B. DVB-T- oder UKW-Rundfunk-Masten) als Sender. Das System besteht aus zwei Empfangskanälen:
1. **Referenzkanal:** Empfängt das direkte Signal vom Sender ($s_{ref}(t)$).
2. **Überwachungskanal (Surveillance):** Empfängt die Echos ($s_{surv}(t)$), die von Objekten (Flugzeugen, Gebäuden etc.) reflektiert werden.

Die Echos von Flugzeugen sind im Vergleich zum direkten Signal und zu Reflexionen vom Boden (Clutter) extrem schwach und zudem vom Hintergrundrauschen überlagert. Darum wird das Prinzip der Signal-Korrelation verwendet, um diese Echos wieder sichtbar zu machen.

---

## 2. Mathematischer Hintergrund der CAF

Die Berechnung, die im Code in `compute_fast_CAF` programmiert wurde, heißt **Kreuzambiguitätsfunktion** (Cross-Ambiguity Function, CAF). Das in der Literatur verwendete mathematische Modell der CAF sieht für die zeitdiskrete Verknüpfung so aus:

$$ \chi(\tau, f_D) = \int_{0}^{T_{CPI}} s_{surv}(t) \cdot s_{ref}^{*}(t - \tau) \cdot e^{-j 2\pi f_D t} dt $$

Wobei:
* $s_{surv}(t)$: Das Überwachungssignal (Echo + Rauschen + Clutter).
* $s_{ref}(t)$: Das Referenzsignal (Direktsignal).
* $s_{ref}^{*}( \dots )$: Die komplex konjugierte Form des Referenzsignals.
* $\tau$: Die Zeitverzögerung (Delay), bestimmt durch die umwegbedingte Laufzeit und steht damit direkt in Bezug zur Entfernung (Range).
* $f_D$: Die Dopplerfrequenz-Verschiebung in Hertz, ausgelöst durch die Bewegung des Ziels.
* $T_{CPI}$: Coherent Processing Interval (Integrationszeit), die Zeit, über die das Signal aufgezeichnet und integriert wird (z. B. $0.1\text{ s}$).

**Einfach ausgedrückt:** Die Formel testet für jede mögliche Verzögerung ($\tau$) und für jede mögliche Geschwindigkeit ($f_D$), ob sich diese Kombination im aufgenommenen Datensalat ("Surveillance") versteckt. 

---

## 3. FFT-basierte, effiziente Berechnung in `analyze_scenario.py`

Da die Auswertung mittels direkter integraler oder summenbasierter Faltung extrem viel Rechenzeit kostet ($O(N^2)$), verwenden wir im Code den Faltungssatz der Fourier-Transformation. Statt jeden Delay $\tau$ einzeln zu verschieben, berechnen wir die komplette Entfernungsachse für eine spezifische Frequenz auf einmal:

1. **DopplerShift anwenden:**
   $$ s^{(f_D)}_{ref}(t) = s_{ref}(t) \cdot e^{j 2 \pi f_D t} $$
2. **Kreuzkorrelation via FFT:**
   Anstelle der Standard-Kreuzkorrelation transformieren wir beide Signale in den Frequenzbereich:
   $$ \text{Corr}(\tau) = \text{IFFT} \Big( \text{FFT}(s_{surv}) \cdot \text{FFT}(s^{(f_D)}_{ref})^* \Big) $$

Dieses Verfahren reduziert den Rechenaufwand von $O(N^2)$ auf $O(N \log N)$ bezogen auf die Samples.

---

## 4. Erklärung der verwendeten Python-/NumPy-Funktionen

Der Code intensiviert die Nutzung von der Python-Bibliothek Numpy (Numerical Python) für schnelle Vektoroperationen:

### Arrays und Matrizen
* **`np.zeros((n, m), dtype=complex)`**: Erzeugt eine Matrix voller Nullen (hier unsere leere Range-Doppler-Map vor der Berechnung).
* **`np.arange(N)`**: Erstellt einen eindimensionalen Array mit den Werten $[0, 1, ..., N-1]$. Wird hier verwendet, um den Zeitvektor `t_vec` zu erstellen.
* **`np.linspace(start, ende, anzahl)`**: Generiert eine lineare Achse. Wird genutzt, um die Doppler-Frequenzachse gleichmäßig aufzubauen (`doppler_freqs`).

### Komplexe Mathematik & Signalverarbeitung
* **`np.exp(...)`**: Wendet die Exponentialfunktion auf jeden Wert in einem Array an. Der Ausdruck `np.exp(1j * 2 * np.pi * f_d * t)` erzeugt die komplexe Trägerwelle (Rotation), mit der wir die Doppler-Hypothese auf das Signal aufmodulieren.
* **`np.conj(...)`**: Bildet das komplex Konjugierte eines Arrays (aus $a + bj$ wird $a - bj$). Ein elementarer Schritt für Korrelationen im Frequenzbereich.
* **`np.fft.fft(...)`**: *Fast Fourier Transform* (Schnelle Fourier-Transformation). Überführt unser zeitdiskretes Signal in das Frequenzspektrum.
* **`np.fft.ifft(...)`**: *Inverse Fast Fourier Transform*. Rechnet ein Array aus dem Frequenzbereich zurück in den Zeitbereich. Das Ergebnis ist unsere Korrelationsachse für alle Laufzeit-Delays.

### Visualisierung (Matplotlib)
* **`plt.imshow(matrix, ...)`**: Zeigt eine 2D-Matrix als Bildraum an. Die Werte der Matrix (CAF-Aplituden) werden in Farben übersetzt (hier die Colormap `jet`, weshalb hohe Werte Rot und Rauschen Blau sind).
* **`vmin` / `vmax`**: Diese Parameter cappen die Darstellungsschwellen. Durch `vmin=-60` und `vmax=0` normieren wir das Bild so, dass alles unter -60 dB (Rauschen) tiefblau gefärbt ist, um Echos (Peaks) besser hervortreten zu lassen.
