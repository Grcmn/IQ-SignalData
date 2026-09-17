"""Orchestrierung: erzeugt blockweise das Arraysignal.
Der Generator hält nur die Objekte zusammen. Die Physik steckt in
geometry/target/scene, die Signalerzeugung in transmitter.
"""

import math

from config import (ARRAY_RADIUS_M, AUDIO_PATH, BLOCK_SIZE, D_OVER_LAMBDA,
                    DELAY_TAPS, DURATION_S, ECHO_AMPLITUDE_STATIC,
                    ENABLE_ECHO, FREQS_HZ, FS_HZ, RX_POS_M, SYSTEM_LOSS_DB,
                    TARGETS, TX_POS_M)
from audio import load_audio
from scene import Scene
from transmitter import DEVIATION_HZ, PILOT_HZ, FMStream
from uca import N_ELEMENTS, radius_from_spacing, wavelength

FC_HZ = FREQS_HZ[0]


def build_scene(fc=FC_HZ, fs=FS_HZ, n=N_ELEMENTS, d_over_lambda=D_OVER_LAMBDA,
                r=ARRAY_RADIUS_M, tx_pos=TX_POS_M, rx_pos=RX_POS_M,
                duration_s=DURATION_S, targets=TARGETS,
                enable_echo=ENABLE_ECHO, n_taps=DELAY_TAPS,
                loss_db=SYSTEM_LOSS_DB,
                static_amplitude=ECHO_AMPLITUDE_STATIC):
    """Baut die Szene auf. Getrennt vom Generator, damit run.py die
    Kennwerte (Aussteuerung, erwartete CAF-Lage) vor dem Lauf abfragen kann."""
    if r is None:
        r = radius_from_spacing(d_over_lambda * wavelength(fc), n)

    return Scene(targets=targets if enable_echo else [],
                 tx=tx_pos, rx=rx_pos, fc=fc, fs=fs,
                 array_radius=r, n_elements=n, duration_s=duration_s,
                 n_taps=n_taps, loss_db=loss_db,
                 static_amplitude=static_amplitude), r


def fm_uca_stream(scene, fs=FS_HZ, audio_path=AUDIO_PATH,
                  duration_s=DURATION_S, deviation_hz=DEVIATION_HZ,
                  pilot_hz=PILOT_HZ, block_size=BLOCK_SIZE, n_blocks=None):
    """Liefert blockweise (n_elements, block_size) 
    """
    if n_blocks is None:
        n_blocks = math.ceil(duration_s * fs / block_size) #anzahl blöcke

    left, right = load_audio(audio_path, fs, duration_s) #laden und vorverarbeitung des audios
    src = FMStream(fs, left, right, deviation_hz=deviation_hz,
                   pilot_hz=pilot_hz)

    for i in range(n_blocks):
        s, phase = src.next_block(block_size)
        if s.size == 0:                       # Audiomaterial erschoepft
            return
        # i * block_size ist der absolute Sampleindex und damit die einzige
        # Uhr des Systems. Siehe TargetEcho.process().
        yield scene.process(s, phase, i * block_size)
