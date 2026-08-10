"""Formatverifikation fuer IQWriterInt16 (.dat + JSON-Sidecar).

Eigenstaendig ausfuehrbar:  python verify_iq_writer.py
Exitcode 0 = alle Pruefungen bestanden, 1 = mindestens eine gescheitert.

Jede Pruefung hat ein hartes Akzeptanzkriterium. Es wird nichts geschaetzt und
nichts "plausibilisiert" -- entweder bitgenau/innerhalb der theoretischen
Schranke oder Fehler.

Der Writer-Code wird von diesem Skript NICHT veraendert und nicht gemockt.
Geschrieben wird immer durch die echte Klasse.
"""

import json
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iq_writer import (FULL_SCALE, SAMPLE_DTYPE, IQWriterInt16, byte_offset,
                       load_iq_int16)
from receiver import receive_array
from uca import azimuth, radius_from_spacing, steering_vector, wavelength

N_ANT = 7
N_FREQ = 16
BYTES_PER_ELEM = 4

RESULTS = []


def record(name, measured, criterion, ok):
    RESULTS.append((name, str(measured), criterion, bool(ok)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {measured}   (Soll: {criterion})")


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def index_coded(n_time, n_freq=N_FREQ, n_ant=N_ANT):
    """(n_freq, n_ant, T) komplex; jeder Wert kodiert seine eigenen Indizes.

    re = t*10000 + f*100 + a,  im = -re
    """
    x = np.zeros((n_freq, n_ant, n_time), dtype=np.complex128)
    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                v = t * 10000 + f * 100 + a
                x[f, a, t] = complex(v, -v)
    return x


def expected_flat(n_time, n_freq=N_FREQ, n_ant=N_ANT):
    """Erwartete flache int16-Folge, per Python-Schleife -- bewusst OHNE numpy-
    reshape, damit der Test nicht dieselbe Annahme wie der Writer benutzt."""
    out = []
    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                v = t * 10000 + f * 100 + a
                out.append(v)       # Realteil
                out.append(-v)      # Imaginaerteil
    return np.array(out, dtype=SAMPLE_DTYPE)


def write_with_permutation(path, x, perm):
    """Nachbau der Writer-Pipeline mit frei waehlbarer Achsenpermutation.

    Nur fuer den Negativtest 1.2b (absichtlich falsches transpose).
    perm=(2,0,1) ist identisch zu dem, was iq_writer.write_multi tut.
    """
    block = np.transpose(x, perm)
    iq = np.stack([block.real, block.imag], axis=-1) * 1.0
    out = np.ascontiguousarray(
        np.clip(np.rint(iq), -FULL_SCALE - 1, FULL_SCALE).astype(SAMPLE_DTYPE))
    with open(path, "wb") as fh:
        out.tofile(fh)


# --------------------------------------------------------------------------
# 1.1 / 1.2 / 1.3 / 1.4 / 1.5  -- auf der indexkodierten Datei
# --------------------------------------------------------------------------

def test_layout_family(tmp):
    n_time = 3
    x = index_coded(n_time)
    path = tmp / "layout.dat"

    with IQWriterInt16(path, N_ANT, n_freq=N_FREQ, scale=1.0) as w:
        w.write_multi(x)

    # ---- 1.1 Dateigroesse -------------------------------------------------
    head("1.1  Dateigroesse")
    size = path.stat().st_size
    want = n_time * N_FREQ * N_ANT * BYTES_PER_ELEM
    print(f"    n_samples={n_time}  n_freq={N_FREQ}  n_ant={N_ANT}")
    print(f"    Recordgroesse pro Zeitschritt = {N_FREQ*N_ANT*BYTES_PER_ELEM} Byte")
    record("1.1 Dateigroesse [Byte]", size, f"== {want}", size == want)

    # ---- 1.2a Layout ------------------------------------------------------
    head("1.2a  Layout mit indexkodierten Werten")
    raw = np.fromfile(path, dtype=SAMPLE_DTYPE)          # roh und flach
    exp = expected_flat(n_time)
    print("    Erste 30 int16-Werte aus der Datei (roh, flach):")
    print("      ist :", " ".join(f"{v:6d}" for v in raw[:30]))
    print("      soll:", " ".join(f"{v:6d}" for v in exp[:30]))
    print("    Lesehilfe: Paare (re, im); re = t*10000 + f*100 + a.")
    print("      Position 0..13  -> t=0 f=0 a=0..6")
    print("      Position 14..27 -> t=0 f=1 a=0..6   (Sprung 0 -> 100)")
    n_diff = int(np.count_nonzero(raw != exp)) if raw.size == exp.size else -1
    record("1.2a abweichende int16-Werte", n_diff, "== 0",
           raw.size == exp.size and n_diff == 0)

    # Zusatz: der Sprung zwischen den Achsen ist explizit sichtbar
    a_step = int(raw[2] - raw[0])       # a: 0 -> 1
    f_step = int(raw[14] - raw[0])      # f: 0 -> 1
    t_step = int(raw[2 * N_FREQ * N_ANT] - raw[0])   # t: 0 -> 1
    print(f"    Inkrement je Achsenschritt: antenna={a_step}, freq={f_step}, time={t_step}")
    record("1.2a Achsen-Inkremente (a,f,t)", (a_step, f_step, t_step),
           "== (1, 100, 10000)", (a_step, f_step, t_step) == (1, 100, 10000))

    # ---- 1.2b Negativtest: vertauschte Achsen muessen auffallen -----------
    head("1.2b  Negativkontrolle: absichtlich falsches transpose")
    good = tmp / "perm_good.dat"
    bad_fa = tmp / "perm_bad_fa.dat"
    bad_tf = tmp / "perm_bad_tf.dat"
    write_with_permutation(good, x, (2, 0, 1))    # korrekt:  t,f,a
    write_with_permutation(bad_fa, x, (2, 1, 0))  # falsch:   t,a,f
    write_with_permutation(bad_tf, x, (0, 2, 1))  # falsch:   f,t,a

    r_good = np.fromfile(good, dtype=SAMPLE_DTYPE)
    r_bad_fa = np.fromfile(bad_fa, dtype=SAMPLE_DTYPE)
    r_bad_tf = np.fromfile(bad_tf, dtype=SAMPLE_DTYPE)
    print("    perm=(2,0,1) [t][f][a] korrekt , erste 16:",
          " ".join(f"{v:6d}" for v in r_good[:16]))
    print("    perm=(2,1,0) [t][a][f] falsch  , erste 16:",
          " ".join(f"{v:6d}" for v in r_bad_fa[:16]))
    print("    perm=(0,2,1) [f][t][a] falsch  , erste 16:",
          " ".join(f"{v:6d}" for v in r_bad_tf[:16]))
    detects = (np.array_equal(r_good, exp)
               and not np.array_equal(r_bad_fa, exp)
               and not np.array_equal(r_bad_tf, exp))
    record("1.2b Test erkennt vertauschte Achsen",
           f"korrekt=OK, f<->a erkannt={not np.array_equal(r_bad_fa, exp)}, "
           f"t<->f erkannt={not np.array_equal(r_bad_tf, exp)}",
           "alle drei zutreffend", detects)

    # ---- 1.3 Adressformel gegen Byte-Zugriff ------------------------------
    head("1.3  Adressformel vs. direkter Byte-Zugriff (seek + struct.unpack)")
    arr = raw.reshape(n_time, N_FREQ, N_ANT, 2)
    rng = np.random.default_rng(20260810)
    triples = [(int(rng.integers(n_time)), int(rng.integers(N_FREQ)),
                int(rng.integers(N_ANT))) for _ in range(12)]
    triples += [(0, 0, 0), (n_time - 1, N_FREQ - 1, N_ANT - 1)]
    mism = 0
    with open(path, "rb") as fh:
        for (t, f, a) in triples:
            off = byte_offset(t, f, a, N_FREQ, N_ANT)
            fh.seek(off)
            re_b, im_b = struct.unpack("<hh", fh.read(4))
            re_n, im_n = int(arr[t, f, a, 0]), int(arr[t, f, a, 1])
            ok = (re_b, im_b) == (re_n, im_n)
            mism += (not ok)
            print(f"    (t={t:2d}, f={f:2d}, a={a}) off={off:6d}  "
                  f"seek=({re_b:6d},{im_b:6d})  numpy=({re_n:6d},{im_n:6d})  "
                  f"{'ok' if ok else 'MISMATCH'}")
    record("1.3 Abweichungen bei 14 Tripeln", mism, "== 0", mism == 0)

    # ---- 1.4 Zero-Fill ----------------------------------------------------
    head("1.4  Zero-Fill der unbelegten Frequenzslots")
    p2 = tmp / "single_slot.dat"
    sig = (np.arange(1, 5 * N_ANT + 1).reshape(N_ANT, 5)
           * (1 + 1j)).astype(np.complex128)
    with IQWriterInt16(p2, N_ANT, n_freq=N_FREQ, scale=1.0) as w:
        w.write(sig, freq_index=0)
    a2 = np.fromfile(p2, dtype=SAMPLE_DTYPE).reshape(5, N_FREQ, N_ANT, 2)
    inactive = a2[:, 1:, :, :]
    n_nonzero = int(np.count_nonzero(inactive))
    active_nonzero = int(np.count_nonzero(a2[:, 0, :, :]))
    print(f"    aktiver Slot 0    : {active_nonzero} von {5*N_ANT*2} Werten != 0")
    print(f"    Slots 1..15       : {inactive.size} Werte geprueft")
    record("1.4 Werte != 0 in inaktiven Slots", n_nonzero, "== 0", n_nonzero == 0)

    # ---- 1.5 Endianness ---------------------------------------------------
    head("1.5  Endianness")
    p3 = tmp / "endian.dat"
    with IQWriterInt16(p3, 1, n_freq=1, scale=1.0) as w:
        w.write_multi(np.array([[[1 + 0j, 258 + 0j]]], dtype=np.complex128))
    first8 = p3.read_bytes()[:8]
    print(f"    dtype im Quelltext : {SAMPLE_DTYPE!r}  (byteorder={SAMPLE_DTYPE.byteorder!r})")
    print(f"    Bytes 0..7         : {first8.hex(' ')}")
    print("    Erwartet           : 01 00 00 00 02 01 00 00")
    print("      Wert 1   -> 01 00  (LSB zuerst; big-endian waere 00 01)")
    print("      Wert 258 -> 02 01  (0x0102; big-endian waere 01 02)")
    ok = first8 == bytes.fromhex("0100000002010000")
    record("1.5 Bytefolge fuer re=1 / re=258", first8.hex(" "),
           "== '01 00 00 00 02 01 00 00'", ok)
    record("1.5 dtype-Byteorder im Quelltext", repr(SAMPLE_DTYPE.str),
           "'<i2' (nicht native np.int16)", SAMPLE_DTYPE.str == "<i2")


# --------------------------------------------------------------------------
# 1.6 Quantisierungsfehler
# --------------------------------------------------------------------------

def test_quantisation(tmp):
    head("1.6  Quantisierungsfehler Float -> int16 -> Float")
    rng = np.random.default_rng(7)
    T = 4096
    scale = 2 ** 14
    amp = rng.uniform(0.0, 0.99, size=(N_ANT, T))
    pha = rng.uniform(-np.pi, np.pi, size=(N_ANT, T))
    x = (amp * np.exp(1j * pha)).astype(np.complex128)

    path = tmp / "quant.dat"
    with IQWriterInt16(path, N_ANT, n_freq=N_FREQ, scale=scale) as w:
        w.write(x, freq_index=0)
    back, meta = load_iq_int16(path, freq_index=0)      # (T, n_ant) complex64

    # Referenz in float64 rechnen, damit der complex64-Cast des Readers
    # den Vergleich nicht dominiert.
    raw = np.fromfile(path, dtype=SAMPLE_DTYPE).reshape(T, N_FREQ, N_ANT, 2)
    exact = (raw[:, 0, :, 0] + 1j * raw[:, 0, :, 1]) / scale
    err = np.abs(x.T - exact)
    measured = float(err.max())
    bound = float(np.sqrt(2) * 0.5 / scale)
    print(f"    scale = {scale}  ->  1 LSB entspricht {1/scale:.3e}")
    print(f"    max |x - x_hat|        = {measured:.6e}")
    print(f"    theoretische Schranke  = sqrt(2)*0.5/scale = {bound:.6e}")
    print(f"    RMS-Fehler             = {float(np.sqrt((err**2).mean())):.6e}")
    print(f"    Ausschoepfung          = {100*measured/bound:.2f} % der Schranke")
    record("1.6 max. komplexer Betragsfehler", f"{measured:.6e}",
           f"<= {bound:.6e}", measured <= bound)
    # zusaetzlich: der float32-Reader darf nicht schlechter als 1 LSB sein
    err32 = float(np.abs(x.T - back).max())
    print(f"    max |Fehler| ueber load_iq_int16 (complex64) = {err32:.6e}")
    record("1.6b Fehler ueber load_iq_int16", f"{err32:.6e}",
           f"<= {bound*1.01:.6e} (float32-Reserve)", err32 <= bound * 1.01)


# --------------------------------------------------------------------------
# 1.7 Clipping
# --------------------------------------------------------------------------

def test_clipping(tmp):
    head("1.7  Clipping-Zaehler und Saettigungsverhalten")
    rng = np.random.default_rng(11)
    T = 1024
    x = (rng.uniform(0, 0.99, (N_ANT, T))
         * np.exp(1j * rng.uniform(-np.pi, np.pi, (N_ANT, T))))
    p1 = tmp / "noclip.dat"
    w = IQWriterInt16(p1, N_ANT, n_freq=N_FREQ, scale=2 ** 14)
    w.write(x, freq_index=0)
    w.close()
    n_clip = json.loads(p1.with_suffix(".json").read_text())["n_clipped"]
    print(f"    Testdaten |x|max = {float(np.abs(x).max()):.4f}, scale = 16384")
    print(f"    -> groesster int16-Betrag ~ {float(np.abs(x).max())*16384:.0f} von 32767")
    record("1.7a n_clipped bei regulaeren Daten", n_clip, "== 0", n_clip == 0)

    # absichtliche Uebersteuerung
    p2 = tmp / "clip.dat"
    over = np.array([[3.0 + 0j, -3.0 + 0j, 2.5 - 2.5j]], dtype=np.complex128)
    over = np.repeat(over, N_ANT, axis=0)
    w = IQWriterInt16(p2, N_ANT, n_freq=N_FREQ, scale=2 ** 14)
    w.write(over, freq_index=0)
    w.close()
    m2 = json.loads(p2.with_suffix(".json").read_text())
    a = np.fromfile(p2, dtype=SAMPLE_DTYPE).reshape(3, N_FREQ, N_ANT, 2)
    v_pos = int(a[0, 0, 0, 0])     # 3.0 * 16384 = 49152  -> muss 32767 sein
    v_neg = int(a[1, 0, 0, 0])     # -3.0 * 16384         -> muss -32768 sein
    print(f"    Eingabe  +3.0 * 16384 = 49152   -> Datei: {v_pos}")
    print(f"    Eingabe  -3.0 * 16384 = -49152  -> Datei: {v_neg}")
    print(f"    n_clipped im Sidecar            : {m2['n_clipped']}")
    print("    Kontrolle Wraparound: np.int16(49152 mod 2**16) waere "
          f"{np.int16(np.uint16(49152 % 2**16))} -- tritt hier NICHT auf")
    record("1.7b Saettigung positiv", v_pos, "== 32767 (nicht negativ)",
           v_pos == 32767)
    record("1.7c Saettigung negativ", v_neg, "== -32768", v_neg == -32768)
    # 7 Antennen x (Re von +3.0, Re von -3.0, Re und Im von 2.5-2.5j) = 7*4 = 28
    record("1.7d n_clipped bei Uebersteuerung", m2["n_clipped"],
           f"== {4*N_ANT} (7 Ant. x 4 Komponenten)", m2["n_clipped"] == 4 * N_ANT)


# --------------------------------------------------------------------------
# 1.8 Blockgrenzen
# --------------------------------------------------------------------------

def test_block_boundaries(tmp):
    head("1.8  Blockgrenzen: ein grosser Block vs. viele kleine")
    rng = np.random.default_rng(3)
    T = 5000
    x = (rng.standard_normal((N_ANT, T)) + 1j * rng.standard_normal((N_ANT, T))) * 0.2

    p_one = tmp / "one_block.dat"
    with IQWriterInt16(p_one, N_ANT, n_freq=N_FREQ) as w:
        w.write(x, freq_index=0)

    p_many = tmp / "many_blocks.dat"
    sizes = [1, 2, 4096, 1, 900]           # ungleich gross, Summe 5000
    assert sum(sizes) == T
    with IQWriterInt16(p_many, N_ANT, n_freq=N_FREQ) as w:
        off = 0
        for s in sizes:
            w.write(x[:, off:off + s], freq_index=0)
            off += s

    b1, b2 = p_one.read_bytes(), p_many.read_bytes()
    print(f"    1 Block  a {T} Samples        -> {len(b1)} Byte")
    print(f"    {len(sizes)} Bloecke {sizes} -> {len(b2)} Byte")
    same = b1 == b2
    n_diff = 0 if same else int(np.count_nonzero(
        np.frombuffer(b1, np.uint8) != np.frombuffer(b2, np.uint8)))
    record("1.8 abweichende Bytes", n_diff, "== 0 (bitidentisch)", same)


# --------------------------------------------------------------------------
# 1.9 Sidecar-Konsistenz
# --------------------------------------------------------------------------

def naive_reader(dat_path):
    """Liest die Datei ausschliesslich anhand des Sidecars -- ohne Kenntnis
    von iq_writer.py. Genau das, was die C-Seite auch tun muss."""
    meta = json.loads(Path(dat_path).with_suffix(".json").read_text())
    assert meta["dtype"] == "comp_int16"
    assert meta["component_dtype"] == "int16"
    assert meta["byte_order"] == "little"
    assert meta["array_order"] == "C"
    assert meta["axis_order"] == ["time", "freq", "antenna"]
    dt = np.dtype("<i2") if meta["byte_order"] == "little" else np.dtype(">i2")
    shape = tuple(meta["shape"]) + (2,)
    raw = np.fromfile(dat_path, dtype=dt)
    return raw.reshape(shape), meta


def test_sidecar(tmp):
    head("1.9  Sidecar-Konsistenz")
    rng = np.random.default_rng(5)
    T = 777
    x = (rng.standard_normal((N_ANT, T)) + 1j * rng.standard_normal((N_ANT, T))) * 0.1
    path = tmp / "sidecar.dat"
    with IQWriterInt16(path, N_ANT, n_freq=N_FREQ, meta={"fs_hz": 240e3}) as w:
        for off in range(0, T, 250):
            w.write(x[:, off:off + 250], freq_index=0)

    arr, meta = naive_reader(path)
    size = path.stat().st_size
    want = (meta["n_samples"] * meta["n_freq_slots"] * meta["n_antennas"]
            * meta["bytes_per_element"])
    print(f"    shape           = {meta['shape']}")
    print(f"    n_samples       = {meta['n_samples']}   (tatsaechlich geschrieben: {T})")
    print(f"    n_antennas      = {meta['n_antennas']}")
    print(f"    n_freq_slots    = {meta['n_freq_slots']}")
    print(f"    bytes_per_time_step = {meta['bytes_per_time_step']}")
    print(f"    scale           = {meta['scale']}")
    print(f"    Dateigroesse    = {size} Byte, aus Sidecar berechnet = {want} Byte")
    print(f"    duration_s      = {meta.get('duration_s')}")
    record("1.9a Dateigroesse == Sidecar-Vorhersage", size, f"== {want}", size == want)
    record("1.9b n_samples == geschriebene Samples", meta["n_samples"],
           f"== {T}", meta["n_samples"] == T)
    record("1.9c bytes_per_time_step", meta["bytes_per_time_step"],
           f"== {N_FREQ*N_ANT*4}", meta["bytes_per_time_step"] == N_FREQ * N_ANT * 4)

    ref, _ = load_iq_int16(path, as_float=False)
    same = np.array_equal(arr, ref)
    print(f"    naiver Sidecar-Leser vs. load_iq_int16: shape {arr.shape} vs {ref.shape}")
    record("1.9d Rekonstruktion nur aus Sidecar", "identisch" if same else "abweichend",
           "bitidentisch zu load_iq_int16", same)


# --------------------------------------------------------------------------
# 1.10 Physikalische Plausibilitaet
# --------------------------------------------------------------------------

def test_steering(tmp):
    head("1.10  Phasendifferenzen gegen den Steering-Vektor")
    fc = 89.5e6
    tx, rx = (5_000.0, 3_000.0), (0.0, 0.0)
    lam = wavelength(fc)
    r = radius_from_spacing(0.4 * lam, N_ANT)
    phi = azimuth(tx, rx)
    a = steering_vector(phi, fc, r, N_ANT)

    # Quelle: konstante Einhuellende (FM ist konstantmodulig), zufaellige Phase.
    rng = np.random.default_rng(42)
    T = 4096
    A = 0.8
    ph = np.cumsum(rng.normal(0, 0.05, T))
    s = (A * np.exp(1j * ph)).astype(np.complex64)
    x = receive_array(s, a)                     # (n_ant, T), echte Kette

    scale = 2 ** 14
    path = tmp / "steering.dat"
    with IQWriterInt16(path, N_ANT, n_freq=N_FREQ, scale=scale) as w:
        for off in range(0, T, 1024):
            w.write(x[:, off:off + 1024], freq_index=0)

    raw = np.fromfile(path, dtype=SAMPLE_DTYPE).reshape(T, N_FREQ, N_ANT, 2)
    z = (raw[:, 0, :, 0].astype(np.float64)
         + 1j * raw[:, 0, :, 1].astype(np.float64))       # (T, n_ant), int-Einheiten

    soll = np.angle(a * np.conj(a[0]))                    # (n_ant,)
    dphi = np.angle(z * np.conj(z[:, [0]]))               # (T, n_ant)
    ist = np.angle(np.mean(np.exp(1j * dphi), axis=0))    # Mittelwert ueber t

    # Schranke: jeder Kanal traegt hoechstens asin(eps/|z|) bei, eps = sqrt(2)/2 LSB.
    amp_min = float(np.abs(z).min())
    eps = np.sqrt(2) * 0.5
    bound = 2.0 * float(np.arcsin(min(1.0, eps / amp_min)))
    dev = np.abs(np.angle(np.exp(1j * (dphi - soll))))    # (T, n_ant), gewickelt
    max_dev = float(dev.max())
    spread = float(np.abs(np.angle(np.exp(1j * (dphi - ist)))).max())

    print(f"    Azimut phi = {np.degrees(phi):.4f} deg, r = {r:.4f} m, "
          f"lambda = {lam:.4f} m, d/lambda = 0.4")
    print(f"    |z| min = {amp_min:.1f} int-Einheiten  (A*scale = {A*scale:.0f})")
    print()
    print("    Ant |  Soll [deg] |   Ist [deg] |  |Delta| [deg]")
    print("    ----+-------------+-------------+---------------")
    for n in range(N_ANT):
        d = np.degrees(abs(np.angle(np.exp(1j * (ist[n] - soll[n])))))
        print(f"     {n}  | {np.degrees(soll[n]):11.6f} | {np.degrees(ist[n]):11.6f} "
              f"| {d:13.6f}")
    print()
    print(f"    max. Abweichung ueber alle (t, n): {np.degrees(max_dev):.6f} deg")
    print(f"    Schranke aus Quantisierung        : {np.degrees(bound):.6f} deg")
    print(f"    max. Schwankung ueber die Zeit    : {np.degrees(spread):.6f} deg")
    record("1.10a max |Delta phi| (t,n) [deg]", f"{np.degrees(max_dev):.6f}",
           f"<= {np.degrees(bound):.6f}", max_dev <= bound)
    record("1.10b Zeitkonstanz [deg]", f"{np.degrees(spread):.6f}",
           f"<= {np.degrees(bound):.6f}", spread <= bound)


# --------------------------------------------------------------------------

def main():
    print(f"numpy   {np.__version__}")
    print(f"Python  {sys.version.split()[0]}")
    print(f"Writer  {Path(__import__('iq_writer').__file__).resolve()}")

    with tempfile.TemporaryDirectory(prefix="iqverify_") as td:
        tmp = Path(td)
        test_layout_family(tmp)
        test_quantisation(tmp)
        test_clipping(tmp)
        test_block_boundaries(tmp)
        test_sidecar(tmp)
        test_steering(tmp)

    head("Zusammenfassung")
    w1 = max(len(r[0]) for r in RESULTS)
    w2 = max(len(r[1]) for r in RESULTS)
    w3 = max(len(r[2]) for r in RESULTS)
    print(f"{'Pruefung'.ljust(w1)} | {'gemessen'.ljust(w2)} | "
          f"{'Kriterium'.ljust(w3)} | ok")
    print("-" * w1 + "-+-" + "-" * w2 + "-+-" + "-" * w3 + "-+----")
    for name, meas, crit, ok in RESULTS:
        print(f"{name.ljust(w1)} | {meas.ljust(w2)} | {crit.ljust(w3)} | "
              f"{'JA' if ok else 'NEIN'}")
    n_fail = sum(1 for r in RESULTS if not r[3])
    print()
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} bestanden, {n_fail} gescheitert")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
