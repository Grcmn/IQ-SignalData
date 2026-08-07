
import numpy as np

from stream import (BLOCK_SIZE, FC_HZ, FS_HZ, N, N_BLOCKS, fm_uca_stream)


def main():
    n_samples = N_BLOCKS * BLOCK_SIZE

    print(f"Traegerfrequenz  fc = {FC_HZ/1e6:.1f} MHz")
    print(f"Abtastrate       fs = {FS_HZ/1e3:.0f} kHz")
    print(f"Samples             = {n_samples}")
    print(f"Dauer               = {n_samples/FS_HZ:.3f} s")
    print(f"Bloecke             = {N_BLOCKS} a {BLOCK_SIZE} Samples")
    print()

    blocks = []
    for i, block in enumerate(fm_uca_stream()):
        blocks.append(block)
        print(f"  Block {i+1:3d}/{N_BLOCKS}  {block.shape}  {block.dtype}")

    return np.concatenate(blocks, axis=1)        # (N, N_BLOCKS*BLOCK_SIZE)


if __name__ == "__main__":
    main()