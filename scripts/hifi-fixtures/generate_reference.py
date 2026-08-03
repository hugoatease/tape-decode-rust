#!/usr/bin/env python3
"""Generate stage-by-stage reference fixtures for the hifi-decode Rust port.

Synthesizes a deterministic PAL-VHS-shaped HiFi FM RF signal (two carriers,
each frequency-modulated by a known audio tone), decodes it with the real
`vhsdecode.hifi.HiFiDecode` code (not a reimplementation), and dumps the
input and every intermediate stage's output as .npy so the Rust port can be
checked stage-by-stage against real Python behavior.

Deliberately calls the exact methods `HiFiDecode.block_decode` calls, in the
same order (see HiFiDecode.py:2236-2412 and :2166-2234), rather than calling
`block_decode` itself, so intermediate values that method doesn't expose
(pre-resample demod, post-DC-trim-but-pre-DOC audio) can be captured too.

The whole synthetic buffer is decoded as a single call (no BLOCKSIZE
chunking/overlap) — chunking is a step-6 (concurrency/block-stitching)
concern, orthogonal to per-stage DSP correctness, which is what these
fixtures validate.

Run with the vhs-decode venv, from the vhs-decode repo root (so `vhsdecode`
is importable):

    cd /Users/hugo/vhs-decode
    .venv/bin/python3 /Users/hugo/tape-decode-rust/scripts/hifi-fixtures/generate_reference.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vhs-decode"))

from vhsdecode.hifi.HiFiDecode import HiFiDecode, REAL_DTYPE  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "hifi" / "synthetic_pal_vhs"

# --- Synthetic signal parameters -------------------------------------------------
FS = 8_000_000.0  # RF sample rate (Hz); well above 2*(1.8MHz+150kHz) Nyquist
# 50ms is short (400k RF samples) but still gives 50/100 whole cycles of the
# 1/2kHz test tones and a 48kHz-rate FFT bin width of 20Hz - plenty for
# stage-correctness checks, while keeping the fixture set small enough to
# commit (full-second fixtures were ~280MB; this is ~14MB).
DURATION_S = 0.05

CARRIER_L_HZ = 1_400_000.0
CARRIER_R_HZ = 1_800_000.0
DEVIATION_HZ = 150_000.0
TONE_L_HZ = 1_000.0
TONE_R_HZ = 2_000.0


def fm_modulate(t, carrier_hz, deviation_hz, tone_hz, dt):
    """Continuous-phase FM: integrate instantaneous frequency, not the tone
    directly, so the result is a proper frequency-modulated carrier rather
    than a phase-modulated approximation."""
    audio = np.sin(2 * np.pi * tone_hz * t)
    inst_freq = carrier_hz + deviation_hz * audio
    phase = 2 * np.pi * np.cumsum(inst_freq) * dt
    return np.cos(phase).astype(np.float64), audio.astype(np.float32)


def build_options():
    return {
        "input_rate": FS,
        "input_format_override": None,
        "standard": "p",
        "format": "vhs",
        "preview": False,
        "preview_available": False,
        "demod_type": "quadrature",
        "afe_left_carrier_deviation": 0,
        "afe_right_carrier_deviation": 0,
        "afe_left_carrier": 0,
        "afe_right_carrier": 0,
        "resampler_quality": "high",
        "spectral_nr_amount": 0,
        "head_switching_interpolation": True,
        "doc": "full",
        "enable_expander": True,
        "enable_deemphasis": True,
        "auto_fine_tune": False,
        "bias_guess": False,
        "normalize": False,
        "grc": False,
        "audio_rate": 48000,
        "gain": 1.0,
        "input_file": "-",
        "output_file": "-",
        "mode": "s",
        "threads": 1,
    }


def save(name, array):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}.npy", np.asarray(array))
    print(f"  wrote {name}.npy  shape={np.asarray(array).shape}  dtype={np.asarray(array).dtype}")


def main():
    n = int(FS * DURATION_S)
    t = np.arange(n) / FS
    dt = 1.0 / FS

    rf_l, audio_l_ref = fm_modulate(t, CARRIER_L_HZ, DEVIATION_HZ, TONE_L_HZ, dt)
    rf_r, audio_r_ref = fm_modulate(t, CARRIER_R_HZ, DEVIATION_HZ, TONE_R_HZ, dt)
    rf_data = (0.5 * (rf_l + rf_r)).astype(REAL_DTYPE)

    print("Saving inputs and reference tones...")
    save("rf_input", rf_data)
    save("audio_l_reference_tone", audio_l_ref)
    save("audio_r_reference_tone", audio_r_ref)

    options = build_options()
    decoder = HiFiDecode(options=options, is_main_process=True, bias_guess=False)

    print("Stage: AFE bandpass filter")
    filter_l = decoder.afeL.work(rf_data)
    filter_r = decoder.afeR.work(rf_data)
    save("01_afe_filter_l", filter_l)
    save("01_afe_filter_r", filter_r)

    print("Stage: FM demodulation (pre-resample)")
    demod_l = np.empty(len(filter_l), dtype=REAL_DTYPE, order="C")
    demod_r = np.empty(len(filter_r), dtype=REAL_DTYPE, order="C")
    decoder.fmL.work(filter_l, demod_l)
    decoder.fmR.work(filter_r, demod_r)
    save("02_demod_l", demod_l)
    save("02_demod_r", demod_r)

    print("Stage: audio-rate resample (if_rate -> 192kHz)")
    resampled_l = decoder.audio_resampler_l.resample_chunk(demod_l, True)
    decoder.audio_resampler_l.clear()
    resampled_r = decoder.audio_resampler_r.resample_chunk(demod_r, True)
    decoder.audio_resampler_r.clear()
    save("03_resampled_l", resampled_l)
    save("03_resampled_r", resampled_r)

    print("Stage: cancelDC_trim")
    pre_l = resampled_l.copy()
    pre_r = resampled_r.copy()
    dc_l = HiFiDecode.cancelDC_trim(pre_l, decoder.audio_process_params.pre_trim)
    dc_r = HiFiDecode.cancelDC_trim(pre_r, decoder.audio_process_params.pre_trim)
    save("04_dc_trimmed_l", pre_l)
    save("04_dc_trimmed_r", pre_r)
    with open(OUT_DIR / "04_dc_values.json", "w") as f:
        json.dump({"dc_l": float(dc_l), "dc_r": float(dc_r)}, f, indent=2)

    print("Stage: dropout compensation (in place)")
    doc_l = pre_l.copy()
    doc_r = pre_r.copy()
    HiFiDecode.dropout_compensate(doc_l, doc_r, decoder.audio_process_params)
    save("05_doc_l", doc_l)
    save("05_doc_r", doc_r)

    print("Stage: head-switch interpolation + final resample")
    final_l, _ = HiFiDecode.head_switch_resample(
        doc_l, decoder.audio_process_params, decoder.audio_final_resampler_l, {}, False
    )
    final_r, _ = HiFiDecode.head_switch_resample(
        doc_r, decoder.audio_process_params, decoder.audio_final_resampler_r, {}, False
    )
    save("06_headswitch_final_l", final_l)
    save("06_headswitch_final_r", final_r)

    print("Stage: stereo mix")
    mixed_l, mixed_r = HiFiDecode.mix_for_mode_stereo(
        final_l, final_r, decoder.audio_process_params.decode_mode
    )
    save("07_stereo_mixed_l", mixed_l)
    save("07_stereo_mixed_r", mixed_r)

    metadata = {
        "fs": FS,
        "duration_s": DURATION_S,
        "carrier_l_hz": CARRIER_L_HZ,
        "carrier_r_hz": CARRIER_R_HZ,
        "deviation_hz": DEVIATION_HZ,
        "tone_l_hz": TONE_L_HZ,
        "tone_r_hz": TONE_R_HZ,
        "options": options,
        "audio_rate_intermediate": decoder.audio_rate,
        "audio_final_rate": decoder.audio_final_rate,
        "standard": {
            "LCarrierRef": decoder.standard.LCarrierRef,
            "RCarrierRef": decoder.standard.RCarrierRef,
            "LCarrierDeviation": decoder.standard.LCarrierDeviation,
            "RCarrierDeviation": decoder.standard.RCarrierDeviation,
            "LNotchWidth": decoder.standard.LNotchWidth,
            "RNotchWidth": decoder.standard.RNotchWidth,
        },
        "pre_trim": decoder.audio_process_params.pre_trim,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote metadata.json and all stage fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
