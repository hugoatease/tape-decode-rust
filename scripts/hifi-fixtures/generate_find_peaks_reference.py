#!/usr/bin/env python3
"""Reference fixtures for hifi-decode's `find_peaks` port
(`crates/hifi-decode/src/find_peaks.rs`), which reimplements the subset of
`scipy.signal.find_peaks` that `HiFiDecode.headswitch_detect_peaks` uses
(no Rust equivalent exists). Two cases, matching the two call sites in
`headswitch_detect_peaks`:

  - `sparse_spikes_distance`: `distance=D, width=1` (the primary
    head-switch-interval peak search).
  - `neighbor_search`: `threshold=T, prominence=0.25, distance=1, width=1`
    (the secondary neighboring-noise search around each primary peak).

Run with the vhs-decode venv:

    cd /Users/hugo/vhs-decode
    .venv/bin/python3 /Users/hugo/tape-decode-rust/scripts/hifi-fixtures/generate_find_peaks_reference.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

OUT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "hifi" / "find_peaks"


def save_case(name, x, **kwargs):
    peaks, props = find_peaks(x, **kwargs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}_x.npy", x.astype(np.float32))
    expected = {
        "peaks": [int(p) for p in peaks],
        "left_ips": [float(v) for v in props.get("left_ips", [])],
        "right_ips": [float(v) for v in props.get("right_ips", [])],
        "prominences": [float(v) for v in props.get("prominences", [])],
    }
    with open(OUT_DIR / f"{name}_expected.json", "w") as f:
        json.dump(expected, f, indent=2)
    print(f"{name}: {len(peaks)} peaks")


def main():
    rng = np.random.default_rng(42)

    # Case 1: sparse spikes on a quiet noisy baseline, mimicking head-switch
    # pulses; distance filter is the load-bearing one here.
    n = 2000
    x = np.zeros(n)
    spike_centers = [100, 105, 400, 900, 901, 1500]
    for c in spike_centers:
        amplitude = 3.0 if c not in (105, 901) else 2.0
        for i in range(max(0, c - 5), min(n, c + 6)):
            x[i] += np.exp(-0.5 * ((i - c) / 2.0) ** 2) * amplitude
    x += rng.normal(0, 0.01, n)
    save_case("sparse_spikes_distance", x, distance=50, width=1)

    # Case 2: threshold+prominence+distance=1, mimicking the neighbor
    # search around a detected primary peak.
    n2 = 300
    x2 = np.abs(rng.normal(0, 0.3, n2))
    x2[50] = 5.0
    x2[51] = 4.0
    x2[150] = 3.0
    x2[151] = 0.5
    x2[152] = 2.8
    save_case("neighbor_search", x2, threshold=0.5, prominence=0.25, distance=1, width=1)


if __name__ == "__main__":
    main()
