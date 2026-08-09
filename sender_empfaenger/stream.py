
from audio import load_audio
from receiver import receive_array
from transmitter import DEVIATION_HZ, PILOT_HZ, FMStream
from uca import (N_ELEMENTS, azimuth, radius_from_spacing, steering_vector, wavelength)

FC_HZ = 89.5e6            
TX_POS_M = (5_000.0, 3_000.0)   
AUDIO_PATH = "audio_music.mp3"    

# Basisband
FS_HZ = 240e3             
DURATION_S = 0.6

# Empfaenger
RX_POS_M = (0.0, 0.0)    
N = N_ELEMENTS    
D_OVER_LAMBDA = 0.4
ARRAY_RADIUS_M = None

# Datenstrom
BLOCK_SIZE = 4096
N_BLOCKS = 30

def fm_uca_stream(fc=FC_HZ, fs=FS_HZ, n=N, d_over_lambda=D_OVER_LAMBDA,
                  r=ARRAY_RADIUS_M, tx_pos=TX_POS_M, rx_pos=RX_POS_M,
                  audio_path=AUDIO_PATH, duration_s=DURATION_S,
                  deviation_hz=DEVIATION_HZ, pilot_hz=PILOT_HZ,
                  block_size=BLOCK_SIZE, n_blocks=N_BLOCKS):
    
    lam = wavelength(fc)
    if r is None:
        r = radius_from_spacing(d_over_lambda * lam, n)
    a = steering_vector(azimuth(tx_pos, rx_pos), fc, r, n)

    left, right = load_audio(audio_path, fs, duration_s)
    src = FMStream(fs, left, right, deviation_hz=deviation_hz,
                   pilot_hz=pilot_hz)

    for _ in range(n_blocks):
        s = src.next_block(block_size)    
        if s.size == 0:               
            return
        yield receive_array(s, a)  
        