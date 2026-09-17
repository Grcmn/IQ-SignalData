import math
from datetime import datetime

import numpy as np

from config import (BLOCK_SIZE, DURATION_S, FREQS_HZ, FS_HZ, N_FREQ, OUTPUT_DIR, RX_POS_M, TX_POS_M)
from config_writer import write_config
from geometry import (bistatic_range, bistatic_velocity, delay_samples, doppler_hz)
from iq_writer import (DEAD_ANTENNAS, N_ANTENNAS_TOTAL, IQWriterInt16, scale_for)
from stream import build_scene, fm_uca_stream
from uca import look_angles


def print_expectations(scene, fc, fs, duration_s, scale, n_invalid):
   
    print()
    print(f"    Skalierung float -> int16 : {scale:.1f}")
    print(f"    ungültige Samples am Anfang (Einschwingen): {n_invalid}")
    print(f"    Winkelkonvention: mathematisch, 0 = +x, gegen Uhrzeigersinn")

    phi_tx, theta_tx = look_angles(TX_POS_M, RX_POS_M)
    print(f"    Direktpfad: R = 0 m, f_d = 0 Hz, "
          f"az = {np.degrees(phi_tx):.2f}\u00b0, el = {np.degrees(theta_tx):.2f}\u00b0")

    for tgt in scene.targets:
        print()
        print(f"    Ziel '{tgt.name}':")
        for label, t in (("t = 0 s", 0.0), (f"t = {duration_s:g} s", duration_s)):
            p = tgt.traj.position(t)
            v = tgt.traj.velocity(t)
            R = float(bistatic_range(p, TX_POS_M, RX_POS_M))
            V = float(bistatic_velocity(p, v, TX_POS_M, RX_POS_M))
            phi, theta = look_angles(p, RX_POS_M)

            print(f"      {label:10s}  R = {R:9.2f} m   V = {V:7.2f} m/s   "
                  f"f_d = {doppler_hz(V, fc):8.3f} Hz   "
                  f"tau = {delay_samples(R, fs):7.4f} Samples   "
                  f"az = {np.degrees(phi):6.2f}\u00b0  el = {np.degrees(theta):6.2f}\u00b0")
        amp = tgt._amplitude(tgt.traj.position(0.0))
        print(f"      Amplitude relativ zum Direktpfad: "
              f"{20 * np.log10(amp):.2f} dB  ({amp * scale:.2f} LSB)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now()
    path = OUTPUT_DIR / f"iq_ref_{started.strftime('%Y%m%d_%H%M%S')}.dat"

    scene, array_radius = build_scene()

    # Aussteuerung aus der tatsächlichen Szene ableiten statt fest zu setzen:
    # mit mehreren Zielen kann |X_n| ueber 1 hinausgehen.
    scale = scale_for(scene.peak_amplitude)

    n_blocks = math.ceil(DURATION_S * FS_HZ / BLOCK_SIZE)
    stream = fm_uca_stream(scene, fs=FS_HZ, duration_s=DURATION_S, block_size=BLOCK_SIZE, n_blocks=n_blocks)

    with IQWriterInt16(path, n_freq=N_FREQ, n_antennas=N_ANTENNAS_TOTAL, dead_antennas=DEAD_ANTENNAS, scale=scale) as writer:
        for i, block in enumerate(stream):          # block: (n_live, T)
            writer.write(block)

            if (i + 1) % 25 == 0 or i == 0:
                print(f"  Block {i+1:4d}/{n_blocks}  {block.shape}  "f"{block.dtype}")
        n_written = writer.n_steps
        bytes_per_step = writer.bytes_per_step

    write_config(path, FREQS_HZ, timestamp=started)

    size = path.stat().st_size
    print()
    print(f"geschrieben: {path}")
    print(f"             {n_written} Zeitschritte, {size/1e6:.2f} MB")
    print(f"             Dauer {n_written/FS_HZ:.3f} s bei fs = {FS_HZ/1e3:.0f} kHz")

    expected = n_written * bytes_per_step
    print(f"             erwartet {expected} Byte, tatsächlich {size} Byte -> "
          f"{'OK' if expected == size else 'ABWEICHUNG'}")

    print()
    print(f"    N_FREQ     = {N_FREQ}")
    freq_list = ", ".join(f"{f/1e6:.4f} MHz" for f in FREQS_HZ)
    print(f"    FREQS      = {freq_list}")

    n_invalid = (scene.group_delay + max((t._delay._hist_len for t in scene.targets), default=0)) # erste samples sind leer aufgrund des delays des interpolators
    print_expectations(scene, FREQS_HZ[0], FS_HZ, DURATION_S, scale, n_invalid) 


if __name__ == "__main__":
    main()
