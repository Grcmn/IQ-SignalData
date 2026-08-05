# Sender, Empfaenger und 7-Element-Kreisarray

Das Sendesignal eines UKW-Rundfunksenders (Illuminator of Opportunity)
erzeugen und als kontinuierlichen I/Q-Datenstrom auf einem 7-Element-
Kreisarray (UCA) abbilden. Noch keine Ziele, kein Doppler, kein Rauschen.

## Dateien

Jede Aufgabe genau einmal: eine Signalquelle (`FMStream`), ein
Einstiegspunkt (`run.py`), alle Grafiken in einer Datei (`plots.py`).

| Datei | Inhalt |
|---|---|
| `transmitter.py` | `FMStream` — die einzige Signalquelle: FM-Basisband (Stereo-MPX oder Toene) blockweise mit fortgefuehrtem Phasenzustand |
| `uca.py` | Geometrie des Kreisarrays: Radius aus Elementabstand, Elementwinkel/-positionen, Azimut zum Sender, Steering-Vektor |
| `receiver.py` | `receive_array()` — Kanalmatrix (7, block) fuer den Direktpfad; `fm_demodulate()` — Kontroll-Demodulation |
| `stream.py` | `fm_uca_stream()` — der Generator, plus alle Szenarioparameter. Ohne Ausgaben, ohne matplotlib |
| `plots.py` | `plot_signal()` und `plot_array()` — saemtliche Kontrollgrafiken |
| `run.py` | Einstiegspunkt: Stream erzeugen, Kontrollen ausgeben, optional plotten |

Die Abhaengigkeiten laufen nur in eine Richtung:
`run.py -> plots.py -> stream.py -> {transmitter, receiver, uca}`.
Die Kernmodule sind damit ohne Plot-Abhaengigkeit importierbar.

## Ausfuehren

```
python run.py           # Datenstrom + Kontrollausgaben
python run.py --plot    # zusaetzlich signal.png und array.png
```

## Szenario: wer steht wo

Alle Positionen in Metern, definiert oben in `stream.py`:

```python
TX_POS_M = (5_000.0, 3_000.0)   # Sender
RX_POS_M = (0.0, 0.0)           # Arrayzentrum des Empfaengers
```

- **Empfaenger (du):** Arrayzentrum im Ursprung (0, 0). Die sieben Elemente
  liegen auf einem Kreis mit r = 1,38 m darum. Element 0 sitzt bei
  (1,38 | 0) m auf der +x-Achse, jedes weitere 360/7 = 51,4 Grad gegen den
  Uhrzeigersinn.
- **Sender:** bei (5000 | 3000) m, also 5 km in +x- und 3 km in +y-Richtung.
  Daraus folgen Entfernung R = 5831 m und Azimut **phi_tx = 30,96 Grad**.

```
   array.png (a) zeigt +-1,7 m       reale Lage (nicht massstaeblich)
   +---------------+                 +-------------------------+
   |    o o o      |                 |                     ^TX |
   |   o  +  ->o   |   ist ein       |                         |
   |    o o        |   Zoom auf +    |   + RX ............>    |
   +---------------+                 |         5831 m, 31 Grad |
                                     +-------------------------+
```

Der Sender liegt also rund 3400 Bildbreiten ausserhalb von Panel (a). Der
rote Pfeil dort ist deshalb **kein Ort, sondern eine Richtung**: ein
Einheitsvektor in Richtung phi_tx, willkuerlich auf 1,4 r skaliert, damit
er ins Bild passt. Seine Laenge sagt nichts ueber die Entfernung aus, und
die Welle laeuft ihm *entgegen* — er zeigt dorthin, woher sie kommt.

Genau das ist die Fernfeldannahme: bei 5,8 km Entfernung und 2,8 m Apertur
ist die Wellenfront ueber dem Array praktisch eben. Die Entfernung erzeugt
nur eine gemeinsame Laufzeit und Daempfung auf allen sieben Kanaelen und
faellt aus den Phasendifferenzen heraus. Was uebrig bleibt, haengt **allein
vom Winkel** ab:

```
alpha_n = (2*pi/lambda) * r * cos(phi_n - phi_tx)
```

Folgen daraus, in der Grafik nachpruefbar:

- Sender verschieben (anderer Winkel) -> Pfeil dreht sich, Panel (b)
  verschiebt sich, Elementpositionen bleiben gleich.
- Nur die Entfernung aendern (gleicher Winkel) -> in dieser Stufe aendert
  sich **nichts**.

## Die Grafiken

Beide entstehen nur mit `python run.py --plot`, getrennt nach
Fragestellung: stimmt das Signal, und stimmt die Geometrie?

### `signal.png` — stimmt das Signal?

![Signal](signal.png)

| Panel | Was zu sehen ist | Wozu |
|---|---|---|
| (a) I/Q im Zeitbereich | Real- und Imaginaerteil auf Kanal 0 | zeigt das rohe I/Q, das der ADC liefern wuerde |
| (b) Einhuellende \|x\| | konstante Linie bei 1 | Nachweis reiner Frequenzmodulation — die Information steckt in der Phase, nicht in der Amplitude |
| (c) FM-Spektrum | breitbandiger Block um 0 Hz mit roten Carson-Markern | Bandbreite ~256 kHz, passt zur Carson-Formel 2*(75 kHz + 15 kHz); sichtbar ist auch, dass `fs = 240 kHz` das Band knapp beschneidet |
| (d) Demoduliertes m(t) | Marker bei 19 und 38 kHz | die Kette stimmt: Pilotton und Stereo-Differenzsignal kommen an der richtigen Stelle wieder heraus, die blockweise Phasenfortfuehrung ist verlustfrei |

Zu Panel (d): bei `MODE = "mpx"` (Default) erscheinen der 19-kHz-Pilot und
der Stereoanteil um 38 kHz; bei `MODE = "tones"` stattdessen die vier
Solltoene 440 Hz, 1 kHz, 3,5 kHz und 11 kHz.

### `array.png` — stimmt die Geometrie?

![Array](array.png)

| Panel | Was zu sehen ist | Wozu |
|---|---|---|
| (a) UCA-Geometrie | sieben nummerierte Elemente, roter Richtungspfeil | Kontrolle der Antennengeometrie und der Einfallsrichtung (siehe Abschnitt oben) |
| (b) Phasendifferenz Kanal n gegen Kanal 0 | Kreuze (gemessen) auf Kreisen (Soll) | der Kern der Stufe: die Phasen im erzeugten Datenstrom entsprechen exakt der Geometrie |

Beide Panels zusammen: Element 1 liegt bei 51,4 Grad und damit am
dichtesten an der Pfeilrichtung (31 Grad) — es hat mit alpha_1 = +155,5 Grad
den groessten Phasenvorlauf. Element 4 auf der Gegenseite (205,7 Grad) hat
mit -165,3 Grad den groessten Nachlauf. Panel (b) zeigt diese Werte relativ
zu Kanal 0; Kreuze und Kreise liegen deckungsgleich (Abweichung ~1e-6 Grad,
begrenzt nur durch den dtype `complex64`).

## Der Datenstrom im Detail

Signalmodell des Direktpfads bei ebener Welle (Fernfeld):

```
X_n(t) = a_n * s(t),    a_n = exp(1j * alpha_n),
alpha_n = (2*pi/lambda) * r * cos(phi_n - phi_tx)
```

mit den Elementwinkeln `phi_n = 2*pi*n/7`, dem Arrayradius
`r = d / (2*sin(pi/7))` und dem Azimut `phi_tx = atan2(y_tx, x_tx)`.
Die Amplitude ist an allen Elementen gleich, nur die Phase unterscheidet
sich — genau das wertet spaeter die Richtungsschaetzung aus.

Parameter (Defaults, oben in `stream.py` definiert):
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

`run.py` prueft: Blockform/-dtype, konstante Phasendifferenz zwischen
Kanal 0 und 1 (Sollwert `alpha_1 - alpha_0`), Sprungfreiheit an den
Blockgrenzen, konstante Einhuellende und die verlustfreie Rueckgewinnung
von `m(t)` (relativer RMS-Fehler ~0 %).

Hinweis zur Abtastrate: `fs = 240 kHz` liegt knapp unter der
Carson-Bandbreite (~256 kHz bei 75 kHz Hub) — in `signal.png` Panel (c)
sichtbar. Fuer die Geometriepruefung ohne Belang, fuer spaetere
Kreuzkorrelation besser `fs = 400 kHz`.

## Naechste Schritte

1. Gemeinsame Laufzeit `tau = R/c` und Freiraumdaempfung des Direktpfads
   (Platz dafuer ist in `receiver.receive_array()` markiert)
2. Rauschen und ADC-Quantisierung im Empfaenger
3. Ein bewegtes Ziel -> Echo mit eigener Richtung, Laufzeit und Doppler
   (Ueberwachungskanal)
