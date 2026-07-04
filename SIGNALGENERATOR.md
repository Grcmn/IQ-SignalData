# Signalgenerator für ein Passivradar-Software-in-the-Loop-System

Das Paket `prsim` erzeugt realistische I/Q-Basisbanddaten der beiden
Empfangskanäle eines Passivradars mit **FM-, DAB- und
DVB-T-Empfangsantennen** — genau die Datenströme, die in echter Hardware
hinter dem ADC anliegen würden. Die nachgelagerte
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
| [waveform.py](prsim/waveform.py) | Drei Illuminator-Waveforms: DVB-T (2k-Modus n. EN 300 744: 1705 Träger, 64-QAM, Pilots mit PRBS und 4/3-Boost, TPS, Guard), DAB (Mode I n. EN 300 401: 1536 Träger, pi/4-DQPSK, NULL-Symbol-Rahmenstruktur), FM (Stereo-MPX mit 19-kHz-Pilot, 75 kHz Hub, Programminhalt "music"/"speech") |
| [geometry.py](prsim/geometry.py) | Sender/Empfänger/Ziele als 3D-Objekte, bistatische Laufzeit & Range, Doppler, bistatische Radargleichung, Friis, kTB-Rauschleistung |
| [channel.py](prsim/channel.py) | Zeitvariable fraktionale Verzögerung (Signalmodell oben) |
| [receiver.py](prsim/receiver.py) | AWGN, LO-Phasenrauschen (Wiener-Prozess, gemeinsamer oder getrennter LO), ADC-Quantisierung mit Clipping und Crest-Faktor-Headroom |
| [scenario.py](prsim/scenario.py) | Szenariodefinition (inkl. Illuminator-Wahl), Orchestrierung, Ground-Truth-Export |
| [analysis.py](prsim/analysis.py) | Batch-CAF (Range-Doppler-Map) zur Validierung — gehört konzeptionell zur nachgelagerten Verarbeitung |

## Warum echte Signalstrukturen statt weißem Rauschen?

Jeder Illuminator hat charakteristische Eigenheiten, die die Ambiguity-
Funktion prägen und mit denen echte Verarbeitungsketten umgehen müssen:

- **DVB-T:** Pilotträger und Guard-Intervall erzeugen **deterministische
  Nebenmaxima** — u.a. Doppler-Linien bei ±1/(4·T_S) ≈ ±992 Hz (das
  Scattered-Pilot-Muster wiederholt sich alle 4 Symbole). Diese Linien
  sind in der Validierungsgrafik sichtbar und entsprechen dem
  Literaturwert.
- **DAB:** Die 96-ms-Rahmenstruktur mit NULL-Symbol erzeugt periodische
  Artefakte in der Range-Doppler-Map.
- **FM:** Die Radareigenschaften hängen vom **Programminhalt** ab
  (`fm_content="music"` vs. `"speech"`); zusammen mit dem sehr starken
  Direktsignal (INR ~60 dB) ist FM-PCL sockelbegrenzt — ohne adaptive
  DPI-Filterung (ECA) heben sich Ziele nur wenige dB ab. Genau dieses
  Verhalten reproduziert der Generator.

Ein Detektor, der nur an weißem Rauschen getestet wurde, produziert an
echten Signalen Geisterziele. Nur der Nutzinhalt (Bits/Audio) ist
zufällig — für Radarzwecke irrelevant.

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

```
python run_scenario.py            # DVB-T-Einzelszenario + Validierung
python run_scenario_multiband.py  # dieselbe Szene über FM, DAB und DVB-T
```

`run_scenario.py` erzeugt `ref_channel.npy`, `surv_channel.npy`
(complex64), `scenario_groundtruth.json` (bistatische Range, Doppler, SNR
je Ziel) und validiert über eine Range-Doppler-Map
(`rd_map_validation.png`).

`run_scenario_multiband.py` erzeugt `ref_{fm,dab,dvbt}.npy` /
`surv_{fm,dab,dvbt}.npy` + Ground Truth und die Vergleichsgrafik
`rd_map_multiband.png` — sie zeigt direkt, wie die Range-Auflösung mit
der Signalbandbreite skaliert (~1 km / ~150 m / ~40 m).

Eigene Szenarien: `Transmitter`, `Receiver`, `Target`, `StaticScatterer`
und `Scenario` (mit `illuminator="dvbt" | "dab" | "fm"`) instanziieren,
`generate()` aufrufen. Für DVB-T/DAB ist `fs` durch die Norm festgelegt
(`FS_DVBT`, `FS_DAB`); für FM frei wählbar (>= 300 kHz).

## Sinnvolle Erweiterungen (Ausbaustufen für die Arbeit)

1. **FM-Inhaltsstudie:** `fm_content="speech"` vs. `"music"` — wie stark
   degradiert die Detektionsleistung bei Sprechpausen?
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
