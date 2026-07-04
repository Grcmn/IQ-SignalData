"""
Szenariodefinition und Top-Level-Signalerzeugung.

generate(scenario) liefert die beiden I/Q-Datenströme, die in einem echten
Passivradar aus den zwei Empfangskanälen (nach ADC) kommen würden:

  ref  — Referenzkanal: Antenne auf den Sender gerichtet
         (Direktsignal + Rauschen)
  surv — Überwachungskanal: Antenne in den Beobachtungsraum gerichtet
         (unterdrücktes Direktsignal (DPI) + Zielechos + Clutter + Rauschen)

Dazu die Ground Truth (bistatische Range, Doppler, SNR) jedes Ziels,
um die nachgelagerte Signalverarbeitung validieren zu können.
"""

from dataclasses import dataclass, field
import numpy as np

from .waveform import generate_dvbt, FS_DVBT
from .geometry import (Transmitter, Receiver, Target, StaticScatterer, C0,
                       bistatic_delay, bistatic_range, doppler_hz, baseline,
                       direct_rx_power, target_rx_power, noise_power)
from .channel import synth_path
from .receiver import awgn, adc_quantize, lo_phase_noise


@dataclass
class Scenario:
    tx: Transmitter
    rx: Receiver
    targets: list = field(default_factory=list)
    scatterers: list = field(default_factory=list)
    fs: float = FS_DVBT
    n_samples: int = 2 ** 21        # CPI = n_samples / fs  (~0.23 s)
    seed: int | None = 1

    # Impairments
    adc_bits: int | None = 12       # None = idealer Empfänger (float)
    adc_headroom_db: float = 15.0
    lo_linewidth_hz: float | None = 50.0   # gemeinsamer LO beider Kanäle
    shared_lo: bool = True

    @property
    def cpi(self) -> float:
        return self.n_samples / self.fs


def generate(sc: Scenario) -> dict:
    """Erzeugt Referenz- und Überwachungskanal für ein Szenario."""
    rng = np.random.default_rng(sc.seed)
    N = sc.n_samples
    t = np.arange(N) / sc.fs
    fc = sc.tx.freq_hz

    # ── Sendesignal mit Vorlauf (für Pfade mit Laufzeit > 0) ─────────
    delays = [bistatic_delay(sc.tx, sc.rx, tgt.pos(t[[0, -1]])).max()
              for tgt in sc.targets]
    delays += [bistatic_delay(sc.tx, sc.rx, s.pos[None, :])[0]
               for s in sc.scatterers]
    delays += [baseline(sc.tx, sc.rx) / C0]
    preroll = int(np.ceil(max(delays) * sc.fs)) + 8

    print(f"Erzeuge DVB-T-Sendesignal ({N + preroll} Samples, "
          f"CPI = {sc.cpi * 1e3:.0f} ms) ...")
    tx_sig = generate_dvbt(N + preroll, seed=sc.seed)

    # ── Leistungsbilanz ──────────────────────────────────────────────
    p_noise = noise_power(sc.fs, sc.rx.noise_figure_db)
    tau_direct = baseline(sc.tx, sc.rx) / C0
    p_dir_ref = direct_rx_power(sc.tx, sc.rx, sc.rx.gain_ref_db)
    p_dpi = direct_rx_power(sc.tx, sc.rx, sc.rx.gain_surv_db) \
        * 10 ** (-sc.rx.dpi_suppression_db / 10)

    print(f"Rauschleistung:      {10*np.log10(p_noise)+30:7.1f} dBm")
    print(f"Direktsignal (Ref):  {10*np.log10(p_dir_ref)+30:7.1f} dBm "
          f"(DNR {10*np.log10(p_dir_ref/p_noise):.1f} dB)")
    print(f"DPI (Surv):         {10*np.log10(p_dpi)+30:8.1f} dBm "
          f"(INR {10*np.log10(p_dpi/p_noise):.1f} dB)")

    # ── Referenzkanal ────────────────────────────────────────────────
    ref = synth_path(tx_sig, t, sc.fs, fc, tau_direct, p_dir_ref, preroll, rng)
    ref += awgn(N, p_noise, rng)

    # ── Überwachungskanal ────────────────────────────────────────────
    surv = synth_path(tx_sig, t, sc.fs, fc, tau_direct, p_dpi, preroll, rng)

    ground_truth = []
    t_mid = sc.cpi / 2
    for tgt in sc.targets:
        tau = bistatic_delay(sc.tx, sc.rx, tgt.pos(t))       # zeitvariabel!
        p_echo = target_rx_power(sc.tx, sc.rx, tgt.pos(t_mid),
                                 tgt.rcs_m2, sc.rx.gain_surv_db)
        surv += synth_path(tx_sig, t, sc.fs, fc, tau, p_echo, preroll, rng)

        rb = bistatic_range(sc.tx, sc.rx, tgt.pos(t_mid))[0]
        fd = doppler_hz(sc.tx, sc.rx, tgt, t_mid)
        snr = 10 * np.log10(p_echo / p_noise)
        ground_truth.append({
            "name": tgt.name,
            "bistatic_range_m": float(rb),
            "delay_samples": float(rb / C0 * sc.fs),
            "doppler_hz": float(fd),
            "snr_per_sample_db": float(snr),
            "rcs_m2": tgt.rcs_m2,
        })
        print(f"  Ziel {tgt.name:<16} R_bi = {rb/1e3:6.2f} km, "
              f"f_D = {fd:+7.1f} Hz, SNR/Sample = {snr:+5.1f} dB")

    for s in sc.scatterers:
        tau = bistatic_delay(sc.tx, sc.rx, s.pos[None, :])[0]
        p_cl = target_rx_power(sc.tx, sc.rx, s.pos[None, :],
                               s.rcs_m2, sc.rx.gain_surv_db)
        surv += synth_path(tx_sig, t, sc.fs, fc, tau, p_cl, preroll, rng)

    surv += awgn(N, p_noise, rng)

    # ── LO-Phasenrauschen ────────────────────────────────────────────
    if sc.lo_linewidth_hz:
        pn = lo_phase_noise(N, sc.fs, sc.lo_linewidth_hz, rng)
        ref *= pn
        surv *= pn if sc.shared_lo else lo_phase_noise(N, sc.fs,
                                                       sc.lo_linewidth_hz, rng)

    # ── ADC ──────────────────────────────────────────────────────────
    adc_scale = None
    if sc.adc_bits:
        # Gemeinsamer Skalierungsfaktor beider Kanäle nicht nötig —
        # echte Empfänger haben pro Kanal eigene AGC/Aussteuerung.
        ref, scale_r = adc_quantize(ref, sc.adc_bits, sc.adc_headroom_db)
        surv, scale_s = adc_quantize(surv, sc.adc_bits, sc.adc_headroom_db)
        adc_scale = {"ref": scale_r, "surv": scale_s}

    return {
        "ref": ref,
        "surv": surv,
        "fs": sc.fs,
        "fc": fc,
        "cpi": sc.cpi,
        "ground_truth": ground_truth,
        "adc_scale": adc_scale,
    }
