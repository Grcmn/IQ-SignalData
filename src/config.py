from pathlib import Path

# ---------------------------------------------------------------- Abtastung
FS_HZ = 152_000.0
DURATION_S = 20.0
BLOCK_SIZE = 4096

AUDIO_OFFSET = 1.0

# ------------------------------------------------------------ Sendefrequenz
FREQS_HZ = [89.5e6]
N_FREQ = len(FREQS_HZ)
N_FREQ_SLOTS = 16

PILOT_HZ = 19e3
DEVIATION_HZ = 75e3

# ---------------------------------------------------------------- Geometrie

TX_POS_M = (5_000.0, 3_000.0, 100.0)
RX_POS_M = (0.0, 0.0, 10.0)

D_OVER_LAMBDA = 0.4                 # Annahme
ARRAY_RADIUS_M = None               # None -> aus D_OVER_LAMBDA berechnen

# ------------------------------------------------------------------- Ziele
#   pos       Startposition zum Zeitpunkt t = 0, also beim ersten Sample
#   vel       konstanter Geschwindigkeitsvektor in m/s
#   rcs       bistatischer Rueckstreuquerschnitt in m^2  [M, Abschn. 2.5.6]
#   level_db  Ueberschreibt den physikalischen Pegel. None = Radargleichung.
#             Der physikalische Wert liegt bei rund -93 dB und damit unter
#             einem LSB der int16-Ausgabe; fuer die Entwicklung deshalb
#             zunaechst auf einen sichtbaren Wert setzen.
TARGETS = [
    dict(name="tgt1",
         pos=(0.0, 8376.4, 1000.0),
         vel=(-59.38, 151.09, 0.0),
         rcs=1.0,
         level_db=-100),

    dict(name="tgt2",
         pos=(30000.0, 79000.0, 9000.0),
         vel=(400.0, 300.0, 0.0),
         rcs=1.0,
         level_db=-100),
         
]

ENABLE_ECHO = True

# ------------------------------------------------------------- Echopfad
DELAY_TAPS = 33                     # ungerade, symmetrischer Sinc-Kern
SYSTEM_LOSS_DB = 10.0               # Systemverluste  [M, Abschn. 2.5.6]
ECHO_AMPLITUDE_STATIC = False       # True = Amplitude auf t=0 einfrieren

# --------------------------------------------------------------- Ausgabe
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/home/alkait/bachelorthesis_iq_data_passivradar/output_iq")
AUDIO_PATH = BASE_DIR.parent / "properties" / "synth-pop.mp3"

TARGET_RMS = 0.7 # 8.4