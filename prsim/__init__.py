"""
prsim — Signalgenerator für ein Passivradar-Software-in-the-Loop-System.

Erzeugt realistische I/Q-Basisbanddaten (Referenz- und Überwachungskanal)
für DVB-T-basiertes Passivradar:

- DVB-T-konformes OFDM-Sendesignal (2k-Modus, Piloten, Guard-Intervall)
- Physikalische Szenariogeometrie (Sender, Empfänger, bewegte Ziele)
- Zeitvariable fraktionale Verzögerung -> Doppler & Range-Migration
  entstehen automatisch aus der Geometrie
- Pegel aus bistatischer Radargleichung und kTB-Rauschleistung
- Direct-Path-Interference, statischer Clutter
- Empfänger-Impairments (Phasenrauschen, ADC-Quantisierung)
"""

from .waveform import generate_dvbt, generate_dab, generate_fm, FS_DVBT, FS_DAB
from .geometry import (C0, Transmitter, Receiver, Target, StaticScatterer,
                       bistatic_delay, bistatic_range, doppler_hz,
                       target_rx_power, direct_rx_power, noise_power)
from .channel import frac_delay, synth_path
from .receiver import awgn, adc_quantize, lo_phase_noise
from .scenario import Scenario, generate
from .analysis import batch_caf

__all__ = [
    "generate_dvbt", "generate_dab", "generate_fm", "FS_DVBT", "FS_DAB", "C0",
    "batch_caf",
    "Transmitter", "Receiver", "Target", "StaticScatterer",
    "bistatic_delay", "bistatic_range", "doppler_hz",
    "target_rx_power", "direct_rx_power", "noise_power",
    "frac_delay", "synth_path",
    "awgn", "adc_quantize", "lo_phase_noise",
    "Scenario", "generate",
]
