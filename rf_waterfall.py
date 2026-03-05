"""
RF Signal Waterfall Visualization
Turning invisible signals into something interpretable and aesthetic.

Inspired by Virgil Abloh's philosophy:
- The 3% shift: recontextualizing existing signals into visual form
- Quotation marks: annotating anomalies to reframe perception
- Transparency: showing raw process, not just polished results
- The Tourist mindset: cross-disciplinary curiosity

This project generates synthetic I/Q signal data, transforms it into
waterfall spectrograms, detects anomalies, and produces annotated
visualizations that treat technical data as a creative medium.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal as sig
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. SYNTHETIC I/Q SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_iq_signal(duration_sec=4.0, sample_rate=2.048e6, seed=42):
    """
    Generate synthetic I/Q data that simulates a busy RF environment.
    Includes background noise, continuous carriers, pulsed transmissions,
    frequency-hopping bursts, and deliberate anomalies.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # base noise floor
    iq = rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    iq *= 0.05

    # continuous narrowband carriers (like FM stations or beacons)
    carrier_freqs = [-0.6e6, -0.25e6, 0.1e6, 0.45e6, 0.72e6]
    for freq in carrier_freqs:
        amplitude = rng.uniform(0.3, 0.8)
        phase = rng.uniform(0, 2 * np.pi)
        bw = rng.uniform(5e3, 25e3)
        noise = rng.standard_normal(num_samples) * bw / sample_rate
        carrier = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase + np.cumsum(noise)))
        iq += carrier

    # pulsed / bursty transmissions
    for _ in range(8):
        freq = rng.uniform(-0.9e6, 0.9e6)
        start = rng.uniform(0, duration_sec * 0.7)
        length = rng.uniform(0.05, 0.4)
        amp = rng.uniform(0.4, 1.0)
        mask = (t >= start) & (t < start + length)
        burst = amp * np.exp(1j * 2 * np.pi * freq * t) * mask
        iq += burst

    # frequency hopping signal (like bluetooth or military comms)
    hop_freqs = rng.uniform(-0.8e6, 0.8e6, size=20)
    hop_duration = 0.08
    for i, hf in enumerate(hop_freqs):
        start = 1.0 + i * hop_duration
        mask = (t >= start) & (t < start + hop_duration)
        hop = 0.6 * np.exp(1j * 2 * np.pi * hf * t) * mask
        iq += hop

    # LoRa CSS (Chirp Spread Spectrum) signal
    # LoRa modulation uses repeated upchirps that sweep across a bandwidth,
    # creating the distinctive sawtooth diagonal lines on a waterfall plot.
    lora_center = 0.35e6       # center frequency offset
    lora_bw = 125e3            # 125 kHz bandwidth (standard LoRa)
    lora_sf = 7                # spreading factor 7 (fastest)
    lora_symbol_time = (2 ** lora_sf) / lora_bw  # ~1.024 ms per symbol
    lora_amp = 0.9

    # transmit several LoRa packets at different times
    lora_packet_starts = [0.3, 1.2, 2.0, 3.0, 3.6]
    lora_symbols_per_packet = [24, 36, 18, 30, 20]

    for pkt_start, n_symbols in zip(lora_packet_starts, lora_symbols_per_packet):
        # preamble: 8 identical upchirps
        preamble_symbols = 8
        total_symbols = preamble_symbols + n_symbols

        for sym_idx in range(total_symbols):
            sym_start = pkt_start + sym_idx * lora_symbol_time
            sym_end = sym_start + lora_symbol_time
            sym_mask = (t >= sym_start) & (t < sym_end)

            if not np.any(sym_mask):
                continue

            t_sym = t[sym_mask] - sym_start  # local time within symbol
            normalized = t_sym / lora_symbol_time  # 0 to 1

            if sym_idx < preamble_symbols:
                # preamble: pure upchirp from -bw/2 to +bw/2
                inst_freq = lora_center + lora_bw * (normalized - 0.5)
            else:
                # data symbols: upchirp with random frequency offset (cyclic shift)
                offset = rng.uniform(0, 1)
                shifted = np.mod(normalized + offset, 1.0)
                inst_freq = lora_center + lora_bw * (shifted - 0.5)

            phase = 2 * np.pi * np.cumsum(inst_freq / sample_rate)
            chirp_signal = lora_amp * np.exp(1j * phase)
            iq[sym_mask] += chirp_signal

    # second LoRa channel at a different frequency (like a second device)
    lora_center_2 = -0.55e6
    lora_packet_starts_2 = [0.6, 1.8, 3.2]
    lora_symbols_per_packet_2 = [20, 28, 16]

    for pkt_start, n_symbols in zip(lora_packet_starts_2, lora_symbols_per_packet_2):
        preamble_symbols = 8
        total_symbols = preamble_symbols + n_symbols

        for sym_idx in range(total_symbols):
            sym_start = pkt_start + sym_idx * lora_symbol_time
            sym_end = sym_start + lora_symbol_time
            sym_mask = (t >= sym_start) & (t < sym_end)

            if not np.any(sym_mask):
                continue

            t_sym = t[sym_mask] - sym_start
            normalized = t_sym / lora_symbol_time

            if sym_idx < preamble_symbols:
                inst_freq = lora_center_2 + lora_bw * (normalized - 0.5)
            else:
                offset = rng.uniform(0, 1)
                shifted = np.mod(normalized + offset, 1.0)
                inst_freq = lora_center_2 + lora_bw * (shifted - 0.5)

            phase = 2 * np.pi * np.cumsum(inst_freq / sample_rate)
            chirp_signal = 0.7 * np.exp(1j * phase)
            iq[sym_mask] += chirp_signal

    # ANOMALIES: unusual wideband bursts (the things we want to detect)
    anomalies = []

    # anomaly 1: sudden wideband pulse
    a1_start, a1_end = 0.8, 0.95
    mask1 = (t >= a1_start) & (t < a1_end)
    wideband = 2.0 * rng.standard_normal(num_samples) * mask1
    iq += wideband + 1j * wideband * 0.5
    anomalies.append({"time": (a1_start, a1_end), "freq": "wideband", "label": "WIDEBAND BURST"})

    # anomaly 2: swept tone (chirp)
    a2_start, a2_end = 2.5, 2.9
    mask2 = (t >= a2_start) & (t < a2_end)
    chirp_sig = 1.5 * sig.chirp(t - a2_start, f0=-0.8e6, f1=0.8e6, t1=0.4, method="linear")
    iq += chirp_sig * mask2
    anomalies.append({"time": (a2_start, a2_end), "freq": "swept", "label": "CHIRP SWEEP"})

    # anomaly 3: repeating pattern (digital burst)
    a3_start = 3.3
    pattern = np.tile([1, 1, -1, 1, -1, -1, 1, -1], 500)
    pattern_upsampled = np.repeat(pattern, int(sample_rate / 40000))
    a3_len = len(pattern_upsampled) / sample_rate
    idx_start = int(a3_start * sample_rate)
    idx_end = min(idx_start + len(pattern_upsampled), num_samples)
    iq[idx_start:idx_end] += 1.2 * pattern_upsampled[:idx_end - idx_start]
    anomalies.append({"time": (a3_start, a3_start + a3_len), "freq": "0.0", "label": "DIGITAL BURST"})

    # LoRa annotations (not anomalies, but labeled signals)
    anomalies.append({"time": (0.3, 0.55), "freq": "0.35", "label": "LoRa CSS"})
    anomalies.append({"time": (0.6, 0.9), "freq": "-0.55", "label": "LoRa CSS CH2"})

    return iq, sample_rate, anomalies


# ---------------------------------------------------------------------------
# 2. SPECTROGRAM / WATERFALL COMPUTATION
# ---------------------------------------------------------------------------

def compute_waterfall(iq, sample_rate, fft_size=1024, overlap=512):
    """
    Compute a waterfall spectrogram from I/Q data.
    Returns frequency axis, time axis, and power spectral density matrix.
    """
    freqs, times, Sxx = sig.spectrogram(
        iq,
        fs=sample_rate,
        nperseg=fft_size,
        noverlap=overlap,
        return_onesided=False,
        mode="psd"
    )

    # shift zero frequency to center
    freqs = np.fft.fftshift(freqs)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    return freqs, times, Sxx_db


# ---------------------------------------------------------------------------
# 3. ANOMALY DETECTION
# ---------------------------------------------------------------------------

def detect_anomalies(freqs, times, Sxx_db, threshold_db=3.0):
    """
    Simple anomaly detection: find time-frequency bins that exceed
    the local median by more than threshold_db.
    Returns a binary mask of anomalous regions.
    """
    from scipy.ndimage import median_filter

    # compute local median background
    background = median_filter(Sxx_db, size=(15, 15))
    deviation = Sxx_db - background

    # threshold
    anomaly_mask = deviation > threshold_db

    return anomaly_mask, deviation


# ---------------------------------------------------------------------------
# 4. CUSTOM COLORMAP (Abloh-inspired aesthetic)
# ---------------------------------------------------------------------------

def create_abloh_colormap():
    """
    Custom colormap inspired by the Off-White x Air Jordan 1 "White" colorway.
    Cream/ivory base, ice blue midtones, soft beige/tan warmth,
    with orange-red peak accents.
    """
    colors = [
        (0.96, 0.95, 0.92),    # warm cream/ivory (noise floor - the "sole")
        (0.92, 0.93, 0.91),    # off-white (the upper)
        (0.88, 0.91, 0.93),    # ice blue tint (the translucent swoosh)
        (0.80, 0.86, 0.90),    # light ice blue
        (0.72, 0.80, 0.86),    # deeper ice blue
        (0.85, 0.82, 0.76),    # soft beige/tan (the suede panels)
        (0.78, 0.75, 0.68),    # warm tan
        (0.30, 0.78, 0.82),    # cyan/teal accent (the stitching)
        (0.95, 0.40, 0.15),    # orange-red (the swoosh tag accent)
        (0.98, 0.30, 0.10),    # bright orange peak (the "AIR" text pop)
    ]
    return LinearSegmentedColormap.from_list("ow_aj1_white", colors, N=256)


def create_mono_colormap():
    """
    Monochrome colormap for anomaly overlay, matching AJ1 White palette.
    """
    colors = [
        (0.0, 0.0, 0.0, 0.0),           # transparent
        (0.30, 0.78, 0.82, 0.2),         # faint cyan
        (0.95, 0.40, 0.15, 0.5),         # orange highlight
    ]
    return LinearSegmentedColormap.from_list("anomaly_overlay", colors, N=256)


# ---------------------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------------------

def render_waterfall(freqs, times, Sxx_db, anomaly_mask, anomalies,
                     sample_rate, output_path="rf_waterfall_output.png"):
    """
    Render the final annotated waterfall visualization.
    Treats the spectrogram as both data and visual artifact.
    """

    # Off-White AJ1 "White" palette constants
    BG = "#f5f3ee"              # warm cream background (the sole)
    TEXT_DARK = "#1a1a1a"       # near-black text
    TEXT_MID = "#8a8680"        # warm gray (like the suede panels)
    ICE_BLUE = "#c8dce6"        # translucent swoosh blue
    CYAN = "#4dc8d1"            # stitching cyan accent
    ORANGE = "#f26522"          # swoosh tag orange
    BEIGE = "#d9d2c5"           # soft beige panel color
    CREAM = "#f0ece4"           # off-white cream

    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[0.08, 1.0, 0.25, 0.06],
        width_ratios=[1, 0.03],
        hspace=0.08, wspace=0.03
    )

    cmap = create_abloh_colormap()
    freqs_mhz = freqs / 1e6

    # --- TITLE BLOCK ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")

    ax_title.text(
        0.0, 0.85,
        '"WATERFALL"',
        transform=ax_title.transAxes,
        fontsize=32, fontweight="bold", fontfamily="monospace",
        color=TEXT_DARK, va="top"
    )
    ax_title.text(
        0.0, 0.25,
        "RF SPECTROGRAM  /  I/Q SIGNAL ANALYSIS  /  ANOMALY DETECTION",
        transform=ax_title.transAxes,
        fontsize=10, fontfamily="monospace",
        color=TEXT_MID, va="top"
    )
    ax_title.text(
        1.0, 0.85,
        "TURNING INVISIBLE\nSIGNALS INTO\nSOMETHING VISIBLE",
        transform=ax_title.transAxes,
        fontsize=10, fontfamily="monospace",
        color=ORANGE, va="top", ha="right",
        linespacing=1.5
    )

    # --- MAIN WATERFALL ---
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.set_facecolor(CREAM)

    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 99)

    im = ax_main.pcolormesh(
        times, freqs_mhz, Sxx_db,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="nearest", rasterized=True
    )

    # anomaly contour overlay in cyan (like the stitching detail)
    ax_main.contour(
        times, freqs_mhz, anomaly_mask.astype(float),
        levels=[0.5], colors=[CYAN], linewidths=0.4, alpha=0.4
    )

    # annotate known anomalies with Abloh-style labels
    for anom in anomalies:
        t_mid = (anom["time"][0] + anom["time"][1]) / 2
        label = f'"{anom["label"]}"'

        if anom["freq"] == "wideband":
            y_pos = freqs_mhz[len(freqs_mhz) // 4]
        elif anom["freq"] == "swept":
            y_pos = 0.0
        else:
            y_pos = float(anom["freq"])

        # draw annotation box
        ax_main.annotate(
            label,
            xy=(t_mid, y_pos),
            xytext=(t_mid + 0.3, y_pos + 0.3),
            fontsize=9, fontfamily="monospace", fontweight="bold",
            color=TEXT_DARK,
            arrowprops=dict(
                arrowstyle="-|>",
                color=ORANGE,
                lw=1.5
            ),
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=1.5
            )
        )

        # time bracket markers
        for t_edge in anom["time"]:
            ax_main.axvline(
                x=t_edge, color=CYAN,
                linestyle="--", linewidth=0.6, alpha=0.5
            )

    ax_main.set_ylabel("FREQUENCY (MHz)", fontsize=10, fontfamily="monospace",
                       color=TEXT_DARK, labelpad=10)
    ax_main.set_xlabel("TIME (seconds)", fontsize=10, fontfamily="monospace",
                       color=TEXT_DARK, labelpad=10)
    ax_main.tick_params(colors=TEXT_MID, labelsize=8)
    for spine in ax_main.spines.values():
        spine.set_color(BEIGE)

    # --- COLORBAR ---
    ax_cb = fig.add_subplot(gs[1, 1])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label("POWER (dB)", fontsize=9, fontfamily="monospace",
                 color=TEXT_MID, labelpad=10)
    cb.ax.tick_params(colors=TEXT_MID, labelsize=7)
    cb.outline.set_edgecolor(BEIGE)

    # --- POWER vs TIME (collapsed spectrum) ---
    ax_power = fig.add_subplot(gs[2, 0])
    ax_power.set_facecolor(CREAM)

    mean_power = np.mean(Sxx_db, axis=0)
    max_power = np.max(Sxx_db, axis=0)
    anomaly_density = np.mean(anomaly_mask, axis=0)

    ax_power.fill_between(times, mean_power, alpha=0.2, color=ORANGE)
    ax_power.plot(times, mean_power, color=ORANGE, linewidth=0.8, label="MEAN POWER")
    ax_power.plot(times, max_power, color=CYAN, linewidth=0.5, alpha=0.6, label="PEAK POWER")

    # highlight anomaly regions
    anomaly_times = anomaly_density > 0.02
    ax_power.fill_between(
        times, mean_power.min(), mean_power,
        where=anomaly_times, alpha=0.1, color=CYAN
    )

    ax_power.set_ylabel("dB", fontsize=9, fontfamily="monospace", color=TEXT_DARK)
    ax_power.set_xlabel("TIME (seconds)", fontsize=9, fontfamily="monospace", color=TEXT_DARK)
    ax_power.tick_params(colors=TEXT_MID, labelsize=7)
    ax_power.legend(fontsize=7, loc="upper right",
                    facecolor="white", edgecolor=BEIGE, labelcolor=TEXT_DARK)
    for spine in ax_power.spines.values():
        spine.set_color(BEIGE)

    # --- FOOTER ---
    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.set_facecolor(BG)
    ax_footer.axis("off")

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    footer_left = (
        f"SAMPLE RATE: {sample_rate/1e6:.3f} MHz   |   "
        f"FFT: 1024   |   "
        f"ANOMALIES DETECTED: {len(anomalies)}   |   "
        f"{timestamp}"
    )
    ax_footer.text(
        0.0, 0.5, footer_left,
        transform=ax_footer.transAxes,
        fontsize=8, fontfamily="monospace", color=TEXT_MID, va="center"
    )
    ax_footer.text(
        1.0, 0.8,
        '"EVERYTHING IN QUOTES"',
        transform=ax_footer.transAxes,
        fontsize=8, fontfamily="monospace", color=ORANGE,
        va="center", ha="right"
    )
    ax_footer.text(
        1.0, 0.1,
        'Sarom Leang, "PhD."',
        transform=ax_footer.transAxes,
        fontsize=9, fontfamily="monospace", color=TEXT_DARK,
        va="center", ha="right"
    )

    # UNCLASSIFIED banners (top and bottom)
    fig.text(0.5, 0.995, "UNCLASSIFIED", fontsize=14, fontweight="bold",
             fontfamily="monospace", color="#00A600", ha="center", va="top")
    fig.text(0.5, 0.005, "UNCLASSIFIED", fontsize=14, fontweight="bold",
             fontfamily="monospace", color="#00A600", ha="center", va="bottom")

    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    print('"WATERFALL"')
    print("Turning invisible signals into something visible.\n")

    print("[1/4] Generating synthetic I/Q signal data...")
    iq, sample_rate, anomalies = generate_iq_signal()
    print(f"      {len(iq):,} samples at {sample_rate/1e6:.3f} MHz")

    print("[2/4] Computing waterfall spectrogram...")
    freqs, times, Sxx_db = compute_waterfall(iq, sample_rate)
    print(f"      Shape: {Sxx_db.shape[0]} freq bins x {Sxx_db.shape[1]} time bins")

    print("[3/4] Detecting anomalies...")
    anomaly_mask, deviation = detect_anomalies(freqs, times, Sxx_db)
    anomaly_pct = 100 * np.mean(anomaly_mask)
    print(f"      {anomaly_pct:.2f}% of time-frequency bins flagged")

    print("[4/4] Rendering visualization...")
    output = render_waterfall(freqs, times, Sxx_db, anomaly_mask, anomalies, sample_rate)

    print(f'\n"COMPLETE"')
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
