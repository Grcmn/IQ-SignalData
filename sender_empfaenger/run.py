
from datetime import datetime
from pathlib import Path

import numpy as np

from iq_writer import IQWriterInt16, load_iq_int16
from stream import (BLOCK_SIZE, D_OVER_LAMBDA, FC_HZ, FS_HZ, N, N_BLOCKS,
                    RX_POS_M, TX_POS_M, fm_uca_stream)
from uca import azimuth

OUTPUT_DIR = Path(__file__).resolve().parent / "output_iq"

# Liest die Datei am Ende zurueck und vergleicht. Haelt dafuer alle Bloecke im
# RAM -- fuer lange Aufnahmen ausschalten.
VERIFY = True


def main():
    n_samples = N_BLOCKS * BLOCK_SIZE

    print(f"Traegerfrequenz  fc = {FC_HZ/1e6:.1f} MHz")
    print(f"Abtastrate       fs = {FS_HZ/1e3:.0f} kHz")
    print(f"Samples             = {n_samples}")
    print(f"Dauer               = {n_samples/FS_HZ:.3f} s")
    print(f"Bloecke             = {N_BLOCKS} a {BLOCK_SIZE} Samples")
    print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"iq_ref_{stamp}.dat"

    meta = {
        "channel": "reference",
        "fs_hz": FS_HZ,
        "fc_hz": FC_HZ,
        "block_size": BLOCK_SIZE,
        "tx_pos_m": list(TX_POS_M),
        "rx_pos_m": list(RX_POS_M),
        "azimuth_rad": float(azimuth(TX_POS_M, RX_POS_M)),
        "d_over_lambda": D_OVER_LAMBDA,
    }

    kept = []
    with IQWriterInt16(path, N, meta=meta) as writer:
        for i, block in enumerate(fm_uca_stream()):
            writer.write(block)
            if VERIFY:
                kept.append(block)
            print(f"  Block {i+1:3d}/{N_BLOCKS}  {block.shape}  {block.dtype}")
        written = writer.n_written
        sidecar = writer.sidecar_path

    size_mb = path.stat().st_size / 1e6
    print()
    print(f"geschrieben: {path}")
    print(f"             {written} Samples, {size_mb:.2f} MB, Form "
          f"({written}, 2, {N}) float32 [time][iq][antenna]")
    print(f"Sidecar:     {sidecar.name}")

    if VERIFY and kept:
        original = np.concatenate(kept, axis=1)          # (N, T)
        #back, _ = load_iq_int16(path)                          # (T, N)
        back, _ = load_iq_int16(path, freq_index=0)
        err = float(np.max(np.abs(original - back.T)))
        print(f"Readback:    max |delta| = {err:.3e}  "
              f"{'OK' if err == 0.0 else 'ABWEICHUNG'}")

    return path


if __name__ == "__main__":
    main()
