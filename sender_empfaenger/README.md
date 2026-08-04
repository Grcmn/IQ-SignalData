# Sender, Empfaenger und 7-Element-Kreisarray

Bausteine des Passivradar-Signalgenerators: das Sendesignal eines
UKW-Rundfunksenders (Illuminator of Opportunity) erzeugen und am
Empfaenger abbilden. Noch keine Ziele, kein Doppler, kein Rauschen.

- **Schritt 1** — Grundkette Sender -> idealer Einzelempfaenger (`run.py`)
- **Schritt 2** — Direktpfad auf ein 7-Element-Kreisarray (UCA), als
  kontinuierlicher, blockweiser Datenstrom (`uca_stream.py`)

## Dateien

| Datei | Inhalt |
|---|---|
| `transmitter.py` | `generate_fm()` — FM-Sendesignal am Stueck (Stereo-MPX + FM); `FMStream` — dieselbe Modulation blockweise mit fortgefuehrtem Phasenzustand |
| `uca.py` | Geometrie des Kreisarrays: Radius aus Elementabstand, Elementwinkel/-positionen, Azimut zum Sender, Steering-Vektor |
| `receiver.py` | `receive()` — idealer Direktempfang; `receive_array()` — Kanalmatrix (7, block) fuer den Direktpfad; `fm_demodulate()` — Kontroll-Demodulation |
| `run.py` | Demo Schritt 1: senden, empfangen, demodulieren, plotten |
| `uca_stream.py` | Demo Schritt 2: Generator `fm_uca_stream()` + Kontrollausgaben |

## Ausfuehren

```
python run.py          # Schritt 1, erzeugt sender_empfaenger.png
python uca_stream.py   # Schritt 2, Datenstrom auf der Konsole
```

`run.py` zeigt vier Ansichten (I/Q-Zeitsignal, konstante Einhuellende,
FM-Spektrum ~256 kHz Carson-Bandbreite, demoduliertes MPX mit 19-kHz-Pilot
und Stereoanteil um 38 kHz) und bestaetigt verlustfreie MPX-Rueckgewinnung.

## Schritt 2 im Detail

Signalmodell des Direktpfads bei ebener Welle (Fernfeld):

```
X_n(t) = a_n * s(t),    a_n = exp(1j * alpha_n),
alpha_n = (2*pi/lambda) * r * cos(phi_n - phi_tx)
```

mit den Elementwinkeln `phi_n = 2*pi*n/7`, dem Arrayradius
`r = d / (2*sin(pi/7))` und dem Azimut `phi_tx = atan2(y_tx, x_tx)`.
Die Amplitude ist an allen Elementen gleich, nur die Phase unterscheidet
sich — genau das wertet spaeter die Richtungsschaetzung aus.

Parameter (Defaults, oben in `uca_stream.py` definiert):
`fc = 100 MHz`, `fs = 240 kHz`, `N = 7`, `d = 0.4*lambda`,
`Sender (5000, 3000) m`, `Empfaenger (0, 0)`, `delta_f = 75 kHz`,
`block_size = 4096`.

Verwendung:

```python
for block in fm_uca_stream(n_blocks=20):
    ...  # block.shape == (7, 4096), dtype complex64
```

Ohne `n_blocks` laeuft der Generator unbegrenzt. Die Momentanphase der FM
wird ueber die Blockgrenzen hinweg fortgefuehrt, das modulierende Signal
laeuft auf einer absoluten Zeitachse — der Strom ist damit sprungfrei.

Die Demo prueft: Blockform/-dtype, konstante Phasendifferenz zwischen
Kanal 0 und 1 (Sollwert `alpha_1 - alpha_0`, Abweichung ~1e-6 Grad,
begrenzt nur durch complex64), Sprungfreiheit an den Blockgrenzen und
konstante Einhuellende.

## Naechste Schritte

1. Gemeinsame Laufzeit `tau = R/c` und Freiraumdaempfung des Direktpfads
   (Platz dafuer ist in `receiver.receive_array()` markiert)
2. Rauschen und ADC-Quantisierung im Empfaenger
3. Ein bewegtes Ziel -> Echo mit eigener Richtung, Laufzeit und Doppler
   (Ueberwachungskanal)
