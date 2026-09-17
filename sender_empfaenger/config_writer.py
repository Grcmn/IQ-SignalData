
from datetime import datetime
from pathlib import Path

DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H:%M:%S"
NEWLINE = "\n"
ENCODING = "ascii"

W_INDEX = 6         # Feldbreite hinter "DRX " fuer den laufenden Index
W_LABEL = 23        # Feldbreite fuer "Frequenz"
W_VALUE = 8         # Mindestbreite des Zahlenwerts, rechtsbuendig
DECIMALS = 4        # Nachkommastellen in MHz


def write_config(iq_path, freqs_hz, timestamp=None):

    freqs = [float(f) for f in freqs_hz]

    if timestamp is None:
        timestamp = datetime.now()

    path = Path(iq_path).with_suffix(".txt")

    lines = [f"{timestamp.strftime(DATE_FMT)}, {timestamp.strftime(TIME_FMT)}"]
    lines += [_drx(i, f) for i, f in enumerate(freqs)]

    text = NEWLINE.join(lines) + NEWLINE
    path.write_text(text, encoding=ENCODING, newline="")
    return path


def _drx(index, freq_hz):
    value = f"{freq_hz / 1e6:.{DECIMALS}f}"
    return (f"DRX {index:<{W_INDEX}}"
            f"{'Frequenz':<{W_LABEL}}"
            f"{value:>{W_VALUE}} MHz")
