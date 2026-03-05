# "SIGNAL"

RF Signal Waterfall Visualization & Drone Detection

Turning invisible signals into something visible and interpretable.

Inspired by Virgil Abloh's philosophy — the 3% shift, quotation marks, transparency, and cross-disciplinary curiosity — this project generates synthetic I/Q signal data, transforms it into waterfall spectrograms, detects anomalies, and produces annotated visualizations that treat technical data as a creative medium.

## Scripts

### `rf_waterfall.py` — RF Signal Waterfall

Generates a synthetic RF environment with narrowband carriers, pulsed transmissions, frequency-hopping signals, LoRa CSS (Chirp Spread Spectrum) on two channels, and deliberate anomalies (wideband burst, chirp sweep, digital burst). Produces an annotated waterfall spectrogram with anomaly detection and a power-over-time plot.

```
python rf_waterfall.py
```

Output: `rf_waterfall_output.png`

### `rf_drone_detect.py` — Drone RF Signature Detection

Simulates an RF environment containing three distinct drone signatures alongside background carriers and a LoRa ground station. The visualization includes an annotated waterfall spectrogram, power-over-time plot, and a drone activity timeline panel.

**Simulated Drones:**

- **Drone 1 (DJI-style)** — FHSS control link (200 kHz bandwidth, 80 hops/sec), wideband video downlink (filtered noise block), and periodic telemetry beacons (4 ms pulses at 200 ms intervals)
- **Drone 2 (smaller craft)** — FHSS control link (150 kHz bandwidth, 50 hops/sec) and telemetry beacons (3 ms pulses at 150 ms intervals)
- **Drone 3 (FPV flyby)** — Analog FM video transmission, brief and high-power, simulating a close-range flyby

**Detection Approach:**

- Waterfall spectrogram computed from synthetic I/Q data (512-point FFT)
- Anomaly detection via median-filtered background subtraction (3 dB threshold)
- Drone signals annotated with time-frequency bounding regions
- Activity timeline shows each signal's presence over time

```
python rf_drone_detect.py
```

Output: `rf_drone_output.png`

## Setup

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`

## Author

Sarom Leang, "PhD."
