"""Messungen zur Erklaerung des Speicherlayouts in iq_writer.write_multi.

python explain_iq_layout.py

Reine Diagnose -- schreibt nur in ein temporaeres Verzeichnis, veraendert den
Writer nicht.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iq_writer import FULL_SCALE, SAMPLE_DTYPE

N_FREQ, N_ANT, T = 16, 7, 4096
SCALE = 2 ** 14


def show(label, a):
    print(f"    {label:<34} shape={str(a.shape):<20} strides={str(a.strides):<24} "
          f"C={a.flags['C_CONTIGUOUS']!s:<5} F={a.flags['F_CONTIGUOUS']!s:<5} "
          f"dtype={a.dtype}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1. Strides
head("1  Strides und Speicherlayout: jeder Zwischenschritt der Pipeline")
x = np.zeros((N_FREQ, N_ANT, T), dtype=np.complex128)
show("x  (Eingabe write_multi)", x)
block = np.transpose(x, (2, 0, 1))
show("block = transpose(x,(2,0,1))", block)
show("block.real  (View!)", block.real)
show("block.imag  (View!)", block.imag)
st = np.stack([block.real, block.imag], axis=-1)
show("np.stack([re,im], axis=-1)", st)
iq = st * SCALE
show("... * scale", iq)
rounded = np.rint(iq)
show("np.rint(...)", rounded)
clipped = np.clip(rounded, -FULL_SCALE - 1, FULL_SCALE)
show("np.clip(...)", clipped)
cast = clipped.astype(SAMPLE_DTYPE)
show("...astype('<i2')", cast)
cont = np.ascontiguousarray(cast)
show("np.ascontiguousarray(...)", cont)
print()
print(f"    ascontiguousarray gibt dasselbe Objekt zurueck: {cont is cast}")
print("    -> False: 'cast' ist NICHT C-contiguous, hier wird wirklich kopiert.")
print("    Grund: stack/arithmetik/rint/clip/astype legen ihr Ergebnis per")
print("    order='K' in der Speicheranordnung der Eingabe ab. Die Eingabe ist")
print("    der transponierte View, also bleibt die ganze Kette in der")
print("    (freq, ant, time)-Anordnung. Das transpose ist bis hierhin rein")
print("    logisch; physisch umsortiert wird erst in ascontiguousarray.")
print()
print("    Kontrollrechnung der Strides von 'cont' (Einheit Byte, itemsize=2):")
print(f"      Achse 3 (re/im) : {cont.strides[3]:8d} = 2")
print(f"      Achse 2 (antenna): {cont.strides[2]:8d} = 2*2")
print(f"      Achse 1 (freq)   : {cont.strides[1]:8d} = 2*2*{N_ANT}")
print(f"      Achse 0 (time)   : {cont.strides[0]:8d} = 2*2*{N_ANT}*{N_FREQ} "
      f"= Recordgroesse {N_FREQ*N_ANT*4} Byte")


# ------------------------------------------------------------- 2. transpose
head("2  Das transpose-Tupel: Permutation und Alternativen")
small = np.zeros((3, 4, 5))     # (freq, ant, time) -- unterscheidbare Groessen
print(f"    Ausgangsform (freq, ant, time) = {small.shape}")
for name, perm in [("[time][freq][antenna]  (aktuell)", (2, 0, 1)),
                   ("[time][antenna][freq]", (2, 1, 0)),
                   ("[freq][time][antenna]", (0, 2, 1))]:
    r = np.transpose(small, perm)
    print(f"    perm={perm} -> {name:<34} Ergebnisform {r.shape}")
print()
print("    Lesart: perm[k] = welche Quellachse an Zielposition k landet.")
print("    Quellachsen: 0=freq, 1=antenna, 2=time")


# ------------------------------------------------------------ 3. stack/view
head("3  np.stack(..., axis=-1)  vs.  .view()")
z = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
print(f"    z (complex128)                 = {z}")
print(f"    z.view(np.float64)             = {z.view(np.float64)}   <- funktioniert")
print(f"    z.view('<i2')                  = {z.view('<i2')}")
print("      ^ das sind die Bitmuster der float64-Zahlen, keine Amplituden.")
print()
zc = z.astype(np.complex64)
print(f"    z.astype(complex64).view('<i2')= {zc.view('<i2')}   <- ebenfalls Unsinn")
print()
print("    Warum view hier ausgeschlossen ist:")
print("      view deutet vorhandene Bytes um, ohne sie zu konvertieren.")
print("      Zwischen float64/float32 und int16 liegt aber eine WERT-Umrechnung")
print("      (Skalierung + Rundung + Saettigung), keine Umdeutung.")
print()
b = np.transpose(np.zeros((2, 3, 4), dtype=np.complex128), (2, 0, 1))
print(f"    Zusatzhuerde: block.real ist ein nicht-zusammenhaengender View")
print(f"      block.real.strides = {b.real.strides}, C_CONTIGUOUS = "
      f"{b.real.flags['C_CONTIGUOUS']}")
try:
    b.real.view(np.float32)
except Exception as e:
    print(f"      block.real.view(np.float32) -> {type(e).__name__}: {e}")
print()
print("    view WAERE moeglich, wenn die Daten bereits als int16-Paare im")
print("    Speicher laegen, z.B. arr_int16.view(np.dtype([('re','<i2'),('im','<i2')])).")
iq16 = np.array([[1, -1], [2, -2]], dtype='<i2')
print(f"      Beispiel: {iq16.ravel()} .view([('re','<i2'),('im','<i2')]) = "
      f"{iq16.view(np.dtype([('re', '<i2'), ('im', '<i2')])).ravel()}")


# -------------------------------------------------- 4. ascontiguousarray
head("4  ascontiguousarray: Korrektheit oder nur Laufzeit?")
rng = np.random.default_rng(1)
xr = (rng.standard_normal((N_FREQ, N_ANT, T))
      + 1j * rng.standard_normal((N_FREQ, N_ANT, T))) * 0.3

def pipeline_upto_astype(xin):
    """Exakt die Schritte aus write_multi, aber OHNE ascontiguousarray."""
    blk = np.transpose(xin, (2, 0, 1))
    q = np.stack([blk.real, blk.imag], axis=-1) * SCALE
    return np.clip(np.rint(q), -FULL_SCALE - 1, FULL_SCALE).astype(SAMPLE_DTYPE)

raw_out = pipeline_upto_astype(xr)
print(f"    Pipeline-Ergebnis vor ascontiguousarray:")
print(f"      C_CONTIGUOUS = {raw_out.flags['C_CONTIGUOUS']}, strides = {raw_out.strides}")
print(f"      ascontiguousarray(...) is raw_out : "
      f"{np.ascontiguousarray(raw_out) is raw_out}  -> es wird kopiert")

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)

    def timed(fn, reps=7):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    p_wrap, p_plain = tmp / "wrap.dat", tmp / "plain.dat"

    def w_wrap():
        with open(p_wrap, "wb") as fh:
            np.ascontiguousarray(raw_out).tofile(fh)

    def w_plain():
        with open(p_plain, "wb") as fh:
            raw_out.tofile(fh)

    t_wrap = timed(w_wrap)
    t_plain = timed(w_plain)
    t_copy = timed(lambda: np.ascontiguousarray(raw_out))
    b_wrap, b_plain = p_wrap.read_bytes(), p_plain.read_bytes()

mb = len(b_wrap) / 1e6
print()
print(f"    Blockgroesse: {mb:.2f} MB  ({T} Zeitschritte), bestes von 7 Durchlaeufen")
print(f"      MIT  ascontiguousarray, dann tofile : {t_wrap*1e3:8.3f} ms")
print(f"      OHNE ascontiguousarray, direkt tofile: {t_plain*1e3:8.3f} ms")
print(f"      davon reine Kopierzeit ascontiguousarray: {t_copy*1e3:8.3f} ms")
print(f"      Faktor ohne/mit: {t_plain/t_wrap:.2f}x")
print()
print(f"    Bytes identisch (Korrektheitsfrage): {b_wrap == b_plain}")
n_diff = 0 if b_wrap == b_plain else int(np.count_nonzero(
    np.frombuffer(b_wrap, np.uint8) != np.frombuffer(b_plain, np.uint8)))
print(f"    abweichende Bytes: {n_diff}")
print(f"    Dateigroesse: {len(b_wrap)} == {T*N_FREQ*N_ANT*4} : "
      f"{len(b_wrap) == T*N_FREQ*N_ANT*4}")
print()
print("    ndarray.tofile schreibt laut Doku immer in logischer C-Reihenfolge,")
print("    unabhaengig vom physischen Layout -- deshalb sind die Bytes gleich.")
print("    ascontiguousarray ist hier also reine Laufzeitoptimierung.")


# ------------------------------------------------ 5. rint -> clip -> astype
head("5  np.rint -> np.clip -> astype  gegen astype allein")
vals = np.array([2.4, 2.5, 2.6, 3.5, -2.4, -2.5, -2.6,
                 32767.4, 49152.0, -49152.0], dtype=np.float64)
print("        Wert |  astype allein | rint |  rint->clip->astype")
print("    ---------+----------------+------+--------------------")
for v in vals:
    a = np.array([v])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            only_astype = int(a.astype(SAMPLE_DTYPE)[0])
    r = float(np.rint(a)[0])
    good = int(np.clip(np.rint(a), -FULL_SCALE - 1, FULL_SCALE)
               .astype(SAMPLE_DTYPE)[0])
    flag = ""
    if only_astype != good:
        flag = "  <-- unterschiedlich"
    print(f"    {v:9.1f} | {only_astype:14d} | {r:5.1f} | {good:18d}{flag}")
print()
print("    Zwei getrennte Fehlerquellen:")
print("      * astype schneidet Richtung Null ab (2.6 -> 2), rint rundet (2.6 -> 3)")
print("      * astype bei Ueberlauf ist in NumPy nicht als Saettigung definiert;")
print("        clip erzwingt sie vorher explizit.")
print()
print("    Reihenfolge clip<->rint: clip zuerst waere bei genau 32767.5 falsch --")
print("    clip(32767.5)=32767 ok, aber rint(32767.5)=32768 -> Ueberlauf.")
demo = np.array([32767.5])
print(f"      rint(32767.5) = {float(np.rint(demo)[0])}  -> danach clip -> "
      f"{int(np.clip(np.rint(demo), -32768, 32767).astype(SAMPLE_DTYPE)[0])}")
print()
print(f"    np.rint benutzt Banker's Rounding (round-half-to-even):")
print(f"      rint(2.5) = {float(np.rint(2.5))},  rint(3.5) = {float(np.rint(3.5))}")
