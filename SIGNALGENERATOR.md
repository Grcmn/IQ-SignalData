# Signalgenerator für ein Passivradar-Software-in-the-Loop-System

Das Paket `prsim` erzeugt realistische I/Q-Basisbanddaten der beiden
Empfangskanäle eines DVB-T-Passivradars — genau die Datenströme, die in
echter Hardware hinter dem ADC anliegen würden. Die nachgelagerte
Signalverarbeitung (CAF, Clutter-Unterdrückung, Detektion, Tracking) kann
damit unter kontrollierten, wiederholbaren Bedingungen mit bekannter
Ground Truth getestet werden — das ist der Kern eines
Software-in-the-Loop-Aufbaus.

## Signalmodell

Jeder Empfangspfad (Direktsignal, Zielecho, Clutter) wird im komplexen
Basisband exakt als

```
y(t) = sqrt(P) · s(t − τ(t)) · exp(−j·2π·f_c·τ(t))
```

modelliert. Dabei ist `s(t)` das Sendesignal, `τ(t)` die zeitvariable
Laufzeit aus der Szenariogeometrie und `f_c` die Trägerfrequenz.

**Der entscheidende Punkt:** Der Doppler wird nicht als fester
Frequenzoffset aufmultipliziert, sondern entsteht automatisch aus der
Zeitableitung der Laufzeit, `f_D(t) = −f_c · dτ/dt`. Damit sind
Doppler-Änderungen über das CPI (Beschleunigung, Kurvenflug) und
Range-Migration physikalisch korrekt enthalten — beides Effekte, an denen
naive Simulationen (fester Delay + feste Dopplerfrequenz) vorbeigehen.

Da `τ(t)·fs` nie ganzzahlig ist, wird `s(t − τ)` per kubischer
Lagrange-Interpolation (Farrow-Struktur, 4 Stützstellen) ausgewertet
([channel.py](prsim/channel.py)).

## Module

| Modul | Inhalt |
|---|---|
| [waveform.py](prsim/waveform.py) | DVB-T-Modulator (2k-Modus n. EN 300 744): 1705 Träger, 64-QAM, Scattered/Continual Pilots mit PRBS und 4/3-Boost, TPS-Träger, Guard-Intervall |
| [geometry.py](prsim/geometry.py) | Sender/Empfänger/Ziele als 3D-Objekte, bistatische Laufzeit & Range, Doppler, bistatische Radargleichung, Friis, kTB-Rauschleistung |
| [channel.py](prsim/channel.py) | Zeitvariable fraktionale Verzögerung (Signalmodell oben) |
| [receiver.py](prsim/receiver.py) | AWGN, LO-Phasenrauschen (Wiener-Prozess, gemeinsamer oder getrennter LO), ADC-Quantisierung mit Clipping und Crest-Faktor-Headroom |
| [scenario.py](prsim/scenario.py) | Szenariodefinition, Orchestrierung, Ground-Truth-Export |

## Warum ein echtes DVB-T-Signal statt weißem Rauschen?

Die Pilotträger und das Guard-Intervall von DVB-T erzeugen
**deterministische Nebenmaxima** in der Ambiguity-Funktion (bekanntes
Problem der Passivradar-Literatur, Stichwort *DVB-T ambiguity function
sidelobes*). Ein Detektor, der nur an weißem Rauschen getestet wurde,
produziert an echten Signalen Geisterziele. Der Generator bildet die
Signalstruktur deshalb normnah ab; nur der Nutzdateninhalt ist zufällig
(für Radarzwecke irrelevant).

## Leistungsbilanz statt gesetzter SNR-Werte

Alle Pegel folgen aus der Physik:

- Zielecho: bistatische Radargleichung `P = ERP·G_rx·λ²·σ / ((4π)³·R_tx²·R_rx²)`
- Direktsignal: Friis-Gleichung; im Überwachungskanal um die
  DPI-Unterdrückung (Antennen-Null/Beamforming) abgeschwächt
- Rauschen: `N = k·T₀·B·F`

Typisches Resultat: Zielechos liegen **pro Sample 10–25 dB unter dem
Rauschen** und werden erst durch den Integrationsgewinn der CAF
(`10·log10(N) ≈ 63 dB` bei 2²¹ Samples) sichtbar — wie im echten System.

## Verwendung

```python
python run_scenario.py
```

erzeugt `ref_channel.npy`, `surv_channel.npy` (complex64),
`scenario_groundtruth.json` (bistatische Range, Doppler, SNR je Ziel) und
validiert das Szenario über eine Batch-CAF-Range-Doppler-Map
(`rd_map_validation.png`). Eigene Szenarien: `Transmitter`, `Receiver`,
`Target`, `StaticScatterer` und `Scenario` instanziieren, `generate()`
aufrufen.

## Sinnvolle Erweiterungen (Ausbaustufen für die Arbeit)

1. **Weitere Illuminatoren:** FM-Rundfunk (schmalbandig, inhaltabhängige
   Ambiguity), DAB (OFDM mit Null-Symbol) — nur `waveform.py` erweitern.
2. **Wegpunkt-/Kurven-Trajektorien** statt geradliniger Bewegung
   (nur `Target.pos(t)` austauschen — der Rest bleibt korrekt, weil alles
   aus τ(t) folgt).
3. **Antennendiagramme:** richtungsabhängiger Gewinn statt Skalarwert;
   Mehrkanal-Empfänger (Array) für Beamforming/DOA.
4. **Multipath im Referenzkanal** (verschmutzte Referenz — bekannter
   Degradationseffekt realer Systeme).
5. **Fluktuierende RCS** (Swerling-Modelle) statt konstanter Werte.
6. **Streaming-Ausgabe** (int16-interleaved über TCP/UDP), um echte
   Echtzeit-Verarbeitungsketten zu füttern (Hardware-in-the-Loop-Vorstufe).
