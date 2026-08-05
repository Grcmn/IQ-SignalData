# Sender, Empfaenger und 7-Element-Kreisarray

Diese Stufe erzeugt das Sendesignal eines UKW-Rundfunksenders (Illuminator of
Opportunity) und bildet es als kontinuierlichen I/Q-Datenstrom auf einem
7-Element-Kreisarray (UCA) ab.

**Was hier bewusst noch fehlt:** Laufzeit, Freiraumdaempfung, Rauschen,
Zielecho, Doppler. Der Direktpfad Sender -> Empfaenger ist alles, was
modelliert wird. Die mathematischen Grundlagen der spaeteren Stufen stehen
in [`README_Passivradar_FM_UCA.md`](README_Passivradar_FM_UCA.md); diese
README beschreibt ausschliesslich den Code in diesem Ordner.

---

## 1. Aufbau: welche Datei macht was

| Datei | Verantwortung | Abhaengig von |
|---|---|---|
| `transmitter.py` | `FMStream` — die **einzige** Signalquelle. FM-Basisband (Stereo-MPX oder Toene), blockweise mit fortgefuehrtem Phasenzustand | nur numpy |
| `uca.py` | **nur Geometrie**: Radius aus Elementabstand, Elementwinkel/-positionen, Azimut zum Sender, Steering-Vektor | nur numpy |
| `receiver.py` | `receive_array()` — Kanalmatrix (7, block); `fm_demodulate()` — Kontroll-Demodulation | `transmitter` |
| `stream.py` | `fm_uca_stream()` — der Generator, plus **alle Szenarioparameter** | `uca`, `receiver`, `transmitter` |
| `plots.py` | `plot_signal()`, `plot_array()` — saemtliche Grafiken, einziger matplotlib-Import | `uca`, `receiver`, `transmitter` |
| `run.py` | Einstiegspunkt: Stream erzeugen, Kontrollen rechnen, optional plotten | alle |

Tatsaechliche Importrichtung (zyklenfrei, von unten nach oben):

```
                 run.py
                /   |   \
        plots.py    |    stream.py
                \   |   /      |
                 receiver.py   uca.py
                      |
                 transmitter.py
```

`plots.py` importiert **nicht** `stream.py` — es bekommt die fertige
Kanalmatrix und das Geometrie-Dict als Argumente uebergeben. Dadurch sind
alle Kernmodule ohne matplotlib importierbar.

---

## 2. Ausfuehren

```bash
python run.py           # Datenstrom + Kontrollausgaben
python run.py --plot    # zusaetzlich signal.png und array.png
```

Benoetigt nur `numpy` (und `matplotlib` fuer `--plot`). Die PNGs landen
immer neben den Quelldateien, unabhaengig vom Arbeitsverzeichnis.

Als Bibliothek:

```python
from stream import fm_uca_stream, stream_info

geo = stream_info()                          # lambda, r, phi_tx, alpha, a
for block in fm_uca_stream(n_blocks=20):
    ...                                      # (7, 4096), complex64
```

Ohne `n_blocks` laeuft der Generator **unbegrenzt** — er haelt keinen Puffer,
der Speicherbedarf bleibt konstant bei einem Block.

---

## 3. Das Szenario

Alle Parameter stehen an einer Stelle, oben in `stream.py`:

| Parameter | Wert | Bedeutung |
|---|---|---|
| `FC_HZ` | 100 MHz | Traegerfrequenz -> `lambda` = 2,998 m |
| `FS_HZ` | 240 kHz | Abtastrate des Basisbands |
| `N` | 7 | Antennenelemente |
| `D_OVER_LAMBDA` | 0,4 | Elementabstand d = 0,4·lambda = 1,199 m |
| `ARRAY_RADIUS_M` | `None` | r direkt vorgeben; `None` -> aus d berechnen |
| `TX_POS_M` | (5000, 3000) m | Senderposition |
| `RX_POS_M` | (0, 0) m | Arrayzentrum |
| `DEVIATION_HZ` | 75 kHz | nomineller Frequenzhub (aus `transmitter.py`) |
| `BLOCK_SIZE` | 4096 | Samples je Block = 17,07 ms |
| `N_BLOCKS` | 20 | Bloecke der Demo = 341 ms = 81 920 Samples |
| `MODE` | `"mpx"` | Modulationsquelle: `"mpx"` oder `"tones"` |

Daraus abgeleitet:

- Arrayradius **r = 1,382 m**, Apertur 2r = 2,76 m
- Entfernung Sender **R = 5831 m**, Azimut **phi_tx = 30,96 Grad**

```
   array.png (a) zeigt +-1,7 m       reale Lage (nicht massstaeblich)
   +---------------+                 +-------------------------+
   |    o o o      |                 |                     ^TX |
   |   o  +  ->o   |   ist ein       |                         |
   |    o o        |   Zoom auf +    |   + RX ............>    |
   +---------------+                 |         5831 m, 31 Grad |
                                     +-------------------------+
```

Der Sender liegt rund 3400 Bildbreiten ausserhalb von Panel (a). Der rote
Pfeil dort ist deshalb **kein Ort, sondern eine Richtung**: ein Einheits-
vektor in Richtung `phi_tx`, willkuerlich auf 1,4·r skaliert. Seine Laenge
sagt nichts ueber die Entfernung, und die Welle laeuft ihm *entgegen* — er
zeigt dorthin, woher sie kommt.

### Warum die Fernfeldannahme hier gilt

Die Fraunhofer-Grenze ist `2·D²/lambda` mit der Apertur D = 2,76 m:

```
2 · 2,76² / 2,998 = 5,1 m   <<   R = 5831 m
```

Der Sender ist also mehr als tausendmal weiter weg als noetig — die
Wellenfront ist ueber dem Array praktisch eben. Konsequenz: die Amplitude
ist an allen sieben Elementen gleich, und Entfernung und Daempfung wirken
als **gemeinsamer Faktor** auf alle Kanaele. Sie fallen aus den
Phasendifferenzen heraus. Uebrig bleibt etwas, das **allein vom Winkel**
abhaengt — genau das wertet spaeter die Richtungsschaetzung aus.

Nachpruefbar in der Grafik:

- Sender verschieben (anderer Winkel) -> Pfeil dreht sich, Panel (b)
  verschiebt sich, Elementpositionen bleiben gleich.
- Nur die Entfernung aendern (gleicher Winkel) -> in dieser Stufe aendert
  sich **nichts**.

---

## 4. Der Sender: `transmitter.py`

### 4.1 Signalkette

```
Audio (Toene)  ->  Stereo-Multiplex m(t)  ->  Phasenintegration  ->  s(t) = exp(j·phi(t))
```

Das Basisbandsignal ist eine **reine Phasenmodulation mit konstantem Betrag**:

```
s(t) = exp(j·phi(t)),    phi(t) = 2·pi · Integral f(tau) dtau,    f(t) = delta_f · m(t)
```

Diskret umgesetzt in `next_block()`:

```python
phase = self._phase + 2*np.pi * np.cumsum(inst_freq) / self.fs
self._phase = float(phase[-1])          # Zustand fuer den naechsten Block
return np.exp(1j * phase)
```

`np.cumsum(...)/fs` ist die Rechteck-Integration der Momentanfrequenz.
`|s(t)| = 1` exakt, die mittlere Leistung damit ebenfalls 1.

### 4.2 Zwei Zustaende, die ueber Blockgrenzen fortgefuehrt werden muessen

Das ist der Kern der Streaming-Faehigkeit. Beide werden in `FMStream`
gehalten:

| Zustand | Attribut | Ohne Fortfuehrung passiert |
|---|---|---|
| Restphase am Blockende | `self._phase` | Phasensprung an jeder Blockgrenze |
| Position auf der Zeitachse von m(t) | `self._n` | m(t) startet in jedem Block neu |

```python
t = (self._n + np.arange(block_size)) / self.fs   # absolute Zeitachse
self._n += block_size
```

Deshalb ist die Normierung von m(t) auch **fest** und nicht blockweise:
eine Maximum-Normierung pro Block wuerde an jeder Blockgrenze einen Sprung
in der Amplitude von m(t) erzeugen.

### 4.3 Das modulierende Signal m(t)

Zwei Modi:

**`mode="tones"`** — Summe von vier Sinustoenen aus `DEFAULT_TONES`:
440 Hz (1,0), 1 kHz (0,6), 3,5 kHz (0,3), 11 kHz (0,15). Geteilt durch die
Amplitudensumme, also garantiert in [-1, 1]. Einfach und gut nachzurechnen.

**`mode="mpx"`** (Default) — Struktur eines echten UKW-Stereo-Multiplex:

```python
mpx = (0.45 * audio                                    # Mono-Summe L+R, 0..15 kHz
       + 0.09 * sin(2*pi*19e3*t)                       # Pilotton
       + 0.45 * diff * sin(2*pi*38e3*t))               # L-R als DSB um 38 kHz
return mpx / 0.99
```

Die Gewichte 45 / 9 / 45 Prozent entsprechen dem Rundfunkstandard
(45 % Mono, 10 % Pilot, 45 % Stereo-Differenz). Der Nenner 0,99 ist die
Summe der drei Gewichte und damit die **obere Schranke** von |mpx|.

### 4.4 Wichtig: der Spitzenhub ist deutlich kleiner als 75 kHz

Die Normierung garantiert nur `|m| <= 1`. Erreicht wird dieser Wert nie,
weil die Toene nicht gleichzeitig ihr Maximum haben:

| Modus | max\|m(t)\| | Spitzenhub | hoechste Modulationsfrequenz | Carson-BW |
|---|---|---|---|---|
| `tones` | 0,986 | 74,0 kHz | 11 kHz | 170 kHz |
| `mpx` | 0,696 | **52,2 kHz** | 38 + 11 = **49 kHz** | **202 kHz** |

Genau dafuer gibt es die Methoden `peak_deviation_hz()`,
`max_modulation_hz()` und `carson_bandwidth_hz()` — die Werte werden aus
dem tatsaechlichen Signal gemessen, nicht geraten.

> **Korrektur gegenueber einer frueheren Fassung dieser README:** dort stand,
> die Carson-Bandbreite sei `2·(75 + 15) kHz` und liege mit „~256 kHz" ausser-
> halb von `fs = 240 kHz`, das Spektrum werde beschnitten. Beides war falsch.
> `2·(75+15)` ist 180, nicht 256 kHz; und im `mpx`-Modus reicht das Modulations-
> signal bis 49 kHz, nicht 15 kHz. Mit dem gemessenen Spitzenhub ergibt sich
> **202 kHz < 240 kHz** — das Signal passt. Nachgemessen bei vierfacher
> Abtastrate liegen **4,9·10⁻⁶ der Leistung (−53 dB)** ausserhalb von ±120 kHz
> und falten sich zurueck; im `tones`-Modus sind es −118 dB. Fuer diese Stufe
> irrelevant, fuer die spaetere Kreuzkorrelation ist etwas mehr Reserve
> (z. B. `fs = 400 kHz`) trotzdem angenehm.

### 4.5 Vereinfachungen gegenueber echtem UKW

- Als „L−R" dient die Kosinus-Variante derselben Toene, kein unabhaengiger
  zweiter Audiokanal.
- Keine 50-µs-Preemphase.
- Keine RDS-Traeger bei 57 kHz.
- Der Audioinhalt ist stationaer; echtes Programmmaterial hat eine stark
  schwankende Momentanbandbreite (ca. 12 kHz Sprache bis 100 kHz Musik) —
  das ist beim Passivradar der wesentliche Nachteil des FM-Beleuchters.

---

## 5. Die Geometrie: `uca.py`

Dieses Modul kennt **weder Signalform noch Abtastrate** — nur Winkel,
Abstaende und den daraus folgenden Steering-Vektor.

### 5.1 Radius aus dem Elementabstand

Zwei benachbarte Elemente eines UCA spannen den Zentriwinkel `2·pi/N` auf,
die Sehne zwischen ihnen ist `d = 2·r·sin(pi/N)`, also:

```
r = d / (2·sin(pi/N))          ->  1,199 / (2·0,4339) = 1,382 m
```

Ueblich ist `d <= lambda/2`, damit die Richtungsschaetzung eindeutig
bleibt (keine Grating Lobes). Hier `d = 0,4·lambda` — Bedingung erfuellt.

### 5.2 Elementpositionen

```
phi_n = 2·pi·n/N                     (n = 0 auf der +x-Achse, gegen den Uhrzeigersinn)
p_n   = r · (cos phi_n, sin phi_n)
```

Fuer N = 7: 0, 51,43, 102,86, 154,29, 205,71, 257,14, 308,57 Grad.

### 5.3 Phasenlagen und Steering-Vektor

Die Projektion der Elementposition `p_n` auf die Einfallsrichtung
`u = (cos phi_tx, sin phi_tx)` ist `p_n · u = r·cos(phi_n − phi_tx)`.
Dieser Wegunterschied gegenueber dem Arrayzentrum entspricht der Phase:

```
alpha_n = (2·pi/lambda) · r · cos(phi_n − phi_tx)
a_n     = exp(j · alpha_n)                          |a_n| = 1
```

Vorzeichenkonvention: **positiv** heisst Phasen*vorlauf* — dem Sender
zugewandte Elemente sehen die Wellenfront frueher.

Der Vorfaktor ist hier `2·pi·r/lambda = 2,896 rad = 165,95 Grad`. Weil das
unter 180 Grad liegt, ist `alpha_n` ueber alle Richtungen eindeutig —
dieselbe Aussage wie `d <= lambda/2`, nur vom Radius aus gesehen.

Werte fuer `phi_tx = 30,96 Grad`:

| n | phi_n [deg] | phi_n − phi_tx | alpha_n [deg] |
|---|---|---|---|
| 0 | 0,00 | −30,96 | +142,30 |
| 1 | 51,43 | +20,47 | **+155,47** |
| 2 | 102,86 | +71,90 | +51,57 |
| 3 | 154,29 | +123,33 | −91,16 |
| 4 | 205,71 | +174,75 | **−165,25** |
| 5 | 257,14 | +226,18 | −114,90 |
| 6 | 308,57 | +277,61 | +21,97 |

Element **1** liegt mit 51,4 Grad am dichtesten an der Einfallsrichtung
(31 Grad) und hat den groessten Vorlauf; Element **4** auf der Gegenseite
(205,7 Grad, fast genau 180 Grad daneben) den groessten Nachlauf. Das ist
der Plausibilitaetstest fuer die Geometrie, den `array.png` (a)+(b)
zusammen zeigen.

Der Steering-Vektor haengt **nur von der Richtung ab, nicht von der Zeit**
— deshalb wird er einmal berechnet und auf jeden Block angewendet.

---

## 6. Der Empfaenger: `receiver.py`

### 6.1 Vorwaertsmodell

```python
def receive_array(s, a, dtype=np.complex64):
    return (a[:, None] * s[None, :]).astype(dtype)
```

Ein aeusseres Produkt: `X_n(t) = a_n · s(t)`, Form `(N, M)`, eine Zeile je
Antennenelement. Die Kanalmatrix hat damit **Rang 1** — ein einziger
einfallender Pfad. Erst ein Zielecho macht daraus Rang 2, und erst dann
wird die Richtungsschaetzung interessant.

Warum `complex64`: halber Speicher gegenueber `complex128`, entspricht dem,
was reale SDR-Aufzeichnungen liefern. Der Preis ist eine Phasenaufloesung
von rund 10⁻⁶ Grad — siehe die Streuungswerte in der Ausgabe.

Laufzeit `tau = R/c` und Freiraumdaempfung fehlen hier **absichtlich**:
beide wirken auf alle Kanaele gleich und aendern ohne Rauschen und ohne
zweiten Pfad nichts an den Phasendifferenzen. Sobald das Zielecho dazukommt,
gehoeren sie an genau diese Stelle — als `amp · s(t − tau)` ueber eine
fraktionale Verzoegerung. Die Stelle ist im Docstring markiert.

### 6.2 Demodulation (nur zur Kontrolle)

```python
dphase   = np.angle(iq[1:] * np.conj(iq[:-1]))
inst_freq = dphase * fs / (2*np.pi)
return inst_freq / deviation_hz
```

Die Momentanfrequenz ist die Ableitung der Momentanphase. `iq[1:]·conj(iq[:-1])`
bildet die Phasendifferenz aufeinanderfolgender Samples ohne explizites
Phase-Unwrapping.

Zwei Eigenschaften, die man sich merken sollte:

1. **Die Steering-Phase faellt heraus.** `a_n` ist konstant, verschwindet
   also bei der Differenzbildung. `m(t)` kommt auf **jedem** der sieben
   Kanaele identisch zurueck.
2. **Die Ausgabe ist ein Sample kuerzer** als die Eingabe. `run.py`
   vergleicht deshalb mit `m_soll[1:]`.

Bedingung fuer Eindeutigkeit: der Phasenschritt pro Sample muss unter
180 Grad bleiben, also `Spitzenhub < fs/2`. Hier 52,2 kHz gegen 120 kHz —
der gemessene groesste Schritt betraegt 77,4 Grad.

---

## 7. Der Stream: `stream.py`

`fm_uca_stream()` ist ein Generator. Der Ablauf:

```python
# einmalig, zeitunabhaengig:
lam   = wavelength(fc)
r     = radius_from_spacing(d_over_lambda * lam, n)   # falls r nicht vorgegeben
phi_tx = azimuth(tx_pos, rx_pos)
a      = steering_vector(phi_tx, fc, r, n)
src    = FMStream(fs, deviation_hz=deviation_hz, mode=mode)

# je Block:
s = src.next_block(block_size)      # (block_size,)  complex128, Zustand laeuft weiter
yield receive_array(s, a)           # (n, block_size) complex64
```

`stream_info()` liefert dieselbe Geometrie als Dict (`lam`, `r`, `n`,
`phi_tx`, `alpha`, `a`), ohne einen einzigen Sample zu erzeugen — praktisch
fuer Ausgaben und Plots.

---

## 8. Die Kontrollen in `run.py`

`python run.py` erzeugt diese Ausgabe (verifiziert):

```
=== Geometrie ===
  fc            = 100.0 MHz   ->  lambda = 2.998 m
  Elementabstand d = 0.4 * lambda = 1.199 m
  Arrayradius   r = 1.382 m  (N = 7 Elemente)
  Sender bei    (5000.0, 3000.0) m,  Empfaenger bei (0.0, 0.0) m
  Azimut phi_tx = 30.964 deg
  alpha_n [deg] = +142.30, +155.47, +51.57, -91.16, -165.25, -114.90, +21.97

  erwartete Phasendifferenz Kanal 1 - Kanal 0 = +13.1749 deg

=== Datenstrom (20 Bloecke a 4096 Samples) ===
  Block  0: shape=(7, 4096)  dtype=complex64  dphi(1-0) = +13.1749 deg  (Streuung 1.13e-06 deg)
  ...
  Block 19: shape=(7, 4096)  dtype=complex64  dphi(1-0) = +13.1749 deg  (Streuung 1.11e-06 deg)

  max. Abweichung vom Sollwert ueber alle Bloecke: 5.42e-06 deg
  Phasensprung an der Blockgrenze: 6.212 deg  (max. innerhalb eines Blocks: 77.354 deg)
  |X| min/max = 1.0000 / 1.0000  (soll ~1)

=== Demodulations-Kontrolle (Kanal 0) ===
  m(t) rueckgewonnen: relativer RMS-Fehler = 0.0000 %
  Spitzenhub  = 52.2 kHz (nominell 75 kHz)
  Carson-BW   = 202.3 kHz bei fs = 240 kHz
```

Was jede Zeile beweist:

| Kontrolle | Wert | Aussage |
|---|---|---|
| `dphi(1-0)` je Block | +13,1749 Grad, Streuung 10⁻⁶ | Die Phasendifferenz im Datenstrom trifft die Geometrie. `block[1]·conj(block[0])` eliminiert das gemeinsame `s(t)`; uebrig bleibt `a_1·conj(a_0)` — eine **ueber die ganze Zeit konstante** Phase. Die Reststreuung ist reine `complex64`-Rundung. |
| max. Abweichung | 5,4·10⁻⁶ Grad | gilt ueber alle 20 Bloecke, nicht nur den ersten |
| Phasensprung Blockgrenze | 6,2 Grad **gegen** 77,4 Grad innerhalb | Der Sprung an der Naht ist kleiner als ein typischer Schritt *im* Block — die Zustandsfortfuehrung funktioniert. Waere `self._phase` nicht gesichert, staende hier ein beliebiger Wert bis 180 Grad. |
| `|X| min/max` | 1,0000 / 1,0000 | reine FM, konstante Einhuellende |
| RMS-Fehler m(t) | 0,0000 % | siehe Warnung unten |
| Spitzenhub / Carson | 52,2 kHz / 202,3 kHz | das Signal passt in `fs = 240 kHz` |

> **Ehrliche Einordnung des RMS-Fehlers.** Er ist *exakt* null, weil
> `fm_demodulate()` die algebraische Umkehrung der `cumsum`-Integration in
> `next_block()` ist — die beiden heben sich per Konstruktion auf. Der Test
> beweist deshalb **Buchhaltung**: dass die Blockgrenzen sauber sind, dass die
> Zeitachse fortlaeuft, dass die Steering-Phase wirklich herausfaellt. Er
> beweist **nicht**, dass das Signal frei von Aliasing oder physikalisch
> korrekt ist. Dafuer sind die Bandbreitenzeile und Panel (c) da.

---

## 9. Die Grafiken

Beide entstehen nur mit `python run.py --plot`, getrennt nach Fragestellung.

### `signal.png` — stimmt das Signal?

![Signal](signal.png)

| Panel | Was zu sehen ist | Wozu |
|---|---|---|
| (a) I/Q im Zeitbereich | Real- und Imaginaerteil auf Kanal 0, erste 200 Samples (833 µs) | das rohe I/Q, das ein ADC liefern wuerde. Sichtbar variable Schwingungsdauer = variable Momentanfrequenz |
| (b) Einhuellende \|x\| | perfekt flache Linie bei 1 | Nachweis reiner Frequenzmodulation — die Information steckt in der Phase, nicht in der Amplitude |
| (c) FM-Spektrum | breiter Block um 0 Hz, rote Carson-Marker bei ±101 kHz | Die Marker werden aus dem tatsaechlichen Signal berechnet (`carson_bandwidth_hz()`), nicht hart verdrahtet. Das Spektrum faellt vor der Bandkante (±120 kHz) auf −45 dB ab — es passt in `fs`. |
| (d) Demoduliertes m(t) | diskrete Linien, Marker bei 19 und 38 kHz | die Kette stimmt Ende zu Ende |

Panel (d) im Detail — bei `MODE = "mpx"` sind alle Linien vorhersagbar:

```
0,44 / 1 / 3,5 / 11 kHz     Mono-Summe (die vier Audiotoene)
19 kHz                      Pilotton
38 ± {0,44; 1; 3,5; 11} kHz DSB-Seitenbaender: 27; 34,5; 37; 37,56; 38,44; 39; 41,5; 49 kHz
```

Ein **unterdrueckter Traeger genau bei 38 kHz** waere falsch — DSB-SC hat
dort keine Linie, nur die Seitenbaender. Bei `MODE = "tones"` erscheinen
stattdessen nur die vier Solltoene bis 11 kHz.

### `array.png` — stimmt die Geometrie?

![Array](array.png)

| Panel | Was zu sehen ist | Wozu |
|---|---|---|
| (a) UCA-Geometrie | sieben nummerierte Elemente auf r = 1,38 m, roter Richtungspfeil bei 31 Grad | Kontrolle von Antennengeometrie und Einfallsrichtung (siehe Abschnitt 3) |
| (b) Phasendifferenz Kanal n gegen Kanal 0 | orange Kreuze (gemessen) exakt in blauen Kreisen (Soll) | **der Kern dieser Stufe**: die Phasen im erzeugten Datenstrom entsprechen der Geometrie |

Panel (b) zeigt `alpha_n − alpha_0`, in ±180 Grad zurueckgefaltet:
0; +13,2; −90,7; +126,5; +52,5; +102,8; −120,3 Grad. Gemessen wird ueber
`np.angle(np.mean(x[k]·conj(x[0])))` — die Mittelung ueber alle 81 920
Samples ist zulaessig, weil die Differenzphase zeitlich konstant ist.
Deckungsgleich bis ~10⁻⁶ Grad, begrenzt allein durch `complex64`.

---

## 10. Was in dieser Stufe geprueft wurde

Durchgesehen und nachgerechnet: `transmitter.py`, `uca.py`, `receiver.py`,
`stream.py`, `plots.py`, `run.py`. Die Physik und die Zustandsfuehrung ueber
Blockgrenzen sind korrekt. Korrigiert wurden:

| Fund | Wo | Status |
|---|---|---|
| Carson-Bandbreite hart auf `delta_f + 15 kHz` verdrahtet — im `mpx`-Modus (Default) sowohl beim Hub als auch bei der Modulationsfrequenz falsch, Marker standen bei ±90 statt ±101 kHz | `plots.py` | behoben: `FMStream.carson_bandwidth_hz()` misst am realen Signal |
| README behauptete Carson = 256 kHz > `fs` und ein beschnittenes Spektrum; die angegebene Formel `2·(75+15)` ergab ausserdem 180, nicht 256 | README | behoben, siehe Abschnitt 4.4 |
| `run.py` griff auf die private Methode `FMStream._modulation()` zu | `transmitter.py`, `run.py` | `modulation()` ist jetzt oeffentlich |
| `DEVIATION_HZ` doppelt definiert (`transmitter.py` **und** `stream.py`) — zwei Quellen der Wahrheit | `stream.py` | wird jetzt aus `transmitter.py` importiert und nur re-exportiert |
| Grafiken landeten im Arbeitsverzeichnis, nicht neben der README, die sie einbindet | `plots.py` | Pfade relativ zum Modul |
| `receiver.py` ohne Modul-Docstring, `receive_array()`-Docstring begann mit Leerzeile | `receiver.py` | ergaenzt |
| README beschrieb die Importkette als `run -> plots -> stream -> {...}`; `plots.py` importiert `stream.py` gar nicht | README | korrigiert, siehe Abschnitt 1 |

Bewusst **nicht** geaendert: das `+`-Vorzeichen in `element_phases()`,
`complex64` als Ausgabeformat und `fs = 240 kHz`. Alle drei sind
konsistente Konventionen bzw. tragfaehige Entscheidungen, keine Fehler.

---

## 11. Naechste Schritte

1. Gemeinsame Laufzeit `tau = R/c` und Freiraumdaempfung des Direktpfads
   (die Stelle ist in `receiver.receive_array()` markiert)
2. Rauschen und ADC-Quantisierung im Empfaenger
3. Ein bewegtes Ziel -> Echo mit eigener Richtung, Laufzeit und Doppler
   (Ueberwachungskanal). Damit wird die Kanalmatrix Rang 2, und
   Richtungsschaetzung sowie Cross-Ambiguity werden sinnvoll.
4. Vor Schritt 3 die Abtastrate auf `fs = 400 kHz` erhoehen — mehr Reserve
   fuer die Kreuzkorrelation.
