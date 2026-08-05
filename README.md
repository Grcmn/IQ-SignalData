# IQ-SignalData

Signalgenerator für ein Passivradar-Software-in-the-Loop-System
(Bachelorarbeit): erzeugt realistische I/Q-Basisbanddaten der beiden
Empfangskanäle (Referenz + Überwachung) für FM-, DAB- und
DVB-T-basiertes Passivradar. Details: [SIGNALGENERATOR.md](SIGNALGENERATOR.md)

## Projektstruktur

```
sender_empfaenger/          aktueller Stand: FM-Sender -> 7-Element-UCA
  transmitter.py              FM-Basisband, blockweise (FMStream)
  uca.py                      Geometrie des Kreisarrays, Steering-Vektor
  receiver.py                 Direktpfad-Empfang, Kontroll-Demodulation
  stream.py                   IQ-Datenstrom (7, block) + Parameter
  plots.py                    Kontrollgrafiken
  run.py                      Einstiegspunkt (--plot fuer Grafiken)
prsim/                      frueherer Stand: Signalgenerator-Paket
  waveform.py                 Illuminator-Waveforms (DVB-T, DAB, FM)
  geometry.py                 3D-Geometrie, Radargleichung, Rauschbilanz
  channel.py                  zeitvariable fraktionale Verzögerung
  receiver.py                 AWGN, Phasenrauschen, ADC
  scenario.py                 Szenariodefinition + generate()
  analysis.py                 Batch-CAF zur Validierung
run_scenario.py             Demo: DVB-T-Einzelszenario
run_scenario_multiband.py   Demo: gleiche Szene über FM, DAB, DVB-T
output/                     generierte Daten & Plots (nicht versioniert)
legacy/                     frühe Experimente (Vorstudien)
```

## Verwendung

```
python run_scenario.py            # erzeugt output/ref_channel.npy, ...
python run_scenario_multiband.py  # erzeugt output/ref_{fm,dab,dvbt}.npy, ...
```
