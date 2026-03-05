"""
RF Drone Detection Waterfall
Visualizing drone signatures in the electromagnetic spectrum.

Sarom Leang, "PhD."
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal as sig
from scipy.ndimage import median_filter
from datetime import datetime


def add_hopping_signal(iq, t, sample_rate, rng, center, bw, hop_rate,
                       t_start, t_end, amplitude, duty=0.7):
    """Add FHSS signal by building a continuous frequency trajectory."""
    mask = (t >= t_start) & (t < t_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return

    t_local = t[idx] - t_start
    duration = t_end - t_start
    hop_dur = 1.0 / hop_rate
    num_hops = int(np.ceil(duration * hop_rate))

    # generate hop frequencies
    hop_freqs = center + rng.uniform(-bw/2, bw/2, size=num_hops)

    # build instantaneous frequency for each sample
    hop_indices = (t_local / hop_dur).astype(int)
    hop_indices = np.clip(hop_indices, 0, num_hops - 1)
    inst_freq = hop_freqs[hop_indices]

    # duty cycle mask
    phase_in_hop = (t_local % hop_dur) / hop_dur
    on_mask = phase_in_hop < duty

    phase = 2 * np.pi * np.cumsum(inst_freq / sample_rate)
    signal = amplitude * np.exp(1j * phase) * on_mask
    iq[idx] += signal


def add_telemetry(iq, t, sample_rate, freq, t_start, t_end,
                  interval, pulse_len, amplitude):
    """Add periodic telemetry pulses."""
    pulse_times = np.arange(t_start, t_end, interval)
    for pt in pulse_times:
        mask = (t >= pt) & (t < pt + pulse_len)
        iq[mask] += amplitude * np.exp(1j * 2 * np.pi * freq * t[mask])


def generate_drone_rf(duration_sec=5.0, sample_rate=1.024e6, seed=77):
    rng = np.random.default_rng(seed)
    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples) / sample_rate

    iq = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) * 0.03

    annotations = []

    # background carriers
    for freq in [-0.35e6, 0.42e6]:
        amp = rng.uniform(0.15, 0.3)
        bw_noise = rng.standard_normal(num_samples) * 5e3 / sample_rate
        iq += amp * np.exp(1j * (2 * np.pi * freq * t + np.cumsum(bw_noise)))

    # --- DRONE 1: DJI-STYLE ---

    # FHSS control link
    add_hopping_signal(iq, t, sample_rate, rng,
                       center=0.15e6, bw=200e3, hop_rate=80,
                       t_start=0.4, t_end=4.5, amplitude=0.5)
    annotations.append({"time": (0.4, 4.5), "freq": "0.15",
                        "label": "DRONE 1 FHSS CTRL", "type": "drone"})

    # video downlink (wideband noise block)
    vid_mask = (t >= 0.6) & (t < 4.3)
    vid_idx = np.where(vid_mask)[0]
    vid_noise = (rng.standard_normal(len(vid_idx)) + 1j * rng.standard_normal(len(vid_idx))) * 0.8
    b = sig.firwin(31, 80e3 / (sample_rate / 2), window="hamming")
    vid_filt = sig.lfilter(b, 1.0, vid_noise)
    vid_center = -0.2e6
    vid_filt *= np.exp(1j * 2 * np.pi * vid_center * t[vid_idx])
    iq[vid_idx] += vid_filt
    annotations.append({"time": (0.6, 4.3), "freq": "-0.2",
                        "label": "DRONE 1 VIDEO DL", "type": "drone"})

    # telemetry beacons
    add_telemetry(iq, t, sample_rate, freq=0.33e6,
                  t_start=0.4, t_end=4.5, interval=0.2,
                  pulse_len=0.004, amplitude=0.7)
    annotations.append({"time": (0.4, 4.5), "freq": "0.33",
                        "label": "DRONE 1 TELEMETRY", "type": "drone"})

    # --- DRONE 2: SMALLER CRAFT ---

    add_hopping_signal(iq, t, sample_rate, rng,
                       center=-0.38e6, bw=150e3, hop_rate=50,
                       t_start=1.8, t_end=4.8, amplitude=0.4)
    annotations.append({"time": (1.8, 4.8), "freq": "-0.38",
                        "label": "DRONE 2 FHSS CTRL", "type": "drone"})

    add_telemetry(iq, t, sample_rate, freq=-0.48e6,
                  t_start=1.8, t_end=4.8, interval=0.15,
                  pulse_len=0.003, amplitude=0.5)
    annotations.append({"time": (1.8, 4.8), "freq": "-0.48",
                        "label": "DRONE 2 TELEMETRY", "type": "drone"})

    # --- DRONE 3: FPV FLYBY (brief, loud analog video) ---

    d3_mask = (t >= 3.0) & (t < 3.8)
    d3_idx = np.where(d3_mask)[0]
    d3_center = 0.38e6
    fm_mod = rng.standard_normal(len(d3_idx)) * 100e3
    d3_phase = 2 * np.pi * np.cumsum((d3_center + fm_mod) / sample_rate)
    iq[d3_idx] += 1.2 * np.exp(1j * d3_phase)
    annotations.append({"time": (3.0, 3.8), "freq": "0.38",
                        "label": "FPV DRONE ANALOG TX", "type": "drone"})

    # --- LoRa (ground context) ---
    lora_center = 0.45e6
    lora_bw = 62.5e3
    lora_symbol_time = (2 ** 7) / lora_bw

    for pkt_start, n_sym in [(0.2, 16), (2.2, 12)]:
        for sym_idx in range(8 + n_sym):
            sym_start = pkt_start + sym_idx * lora_symbol_time
            sym_mask = (t >= sym_start) & (t < sym_start + lora_symbol_time)
            if not np.any(sym_mask):
                continue
            t_sym = t[sym_mask] - sym_start
            norm = t_sym / lora_symbol_time
            if sym_idx < 8:
                inst_f = lora_center + lora_bw * (norm - 0.5)
            else:
                off = rng.uniform(0, 1)
                inst_f = lora_center + lora_bw * (np.mod(norm + off, 1.0) - 0.5)
            phase = 2 * np.pi * np.cumsum(inst_f / sample_rate)
            iq[sym_mask] += 0.6 * np.exp(1j * phase)

    annotations.append({"time": (0.2, 0.7), "freq": "0.45",
                        "label": "LoRa GND STATION", "type": "other"})

    return iq, sample_rate, annotations


def compute_waterfall(iq, sample_rate, fft_size=512, overlap=384):
    freqs, times, Sxx = sig.spectrogram(
        iq, fs=sample_rate, nperseg=fft_size, noverlap=overlap,
        return_onesided=False, mode="psd"
    )
    freqs = np.fft.fftshift(freqs)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return freqs, times, 10 * np.log10(Sxx + 1e-12)


def detect_anomalies(Sxx_db, threshold_db=3.0):
    background = median_filter(Sxx_db, size=(11, 11))
    return (Sxx_db - background) > threshold_db


def create_colormap():
    colors = [
        (0.96, 0.95, 0.92),
        (0.92, 0.93, 0.91),
        (0.88, 0.91, 0.93),
        (0.80, 0.86, 0.90),
        (0.72, 0.80, 0.86),
        (0.85, 0.82, 0.76),
        (0.78, 0.75, 0.68),
        (0.30, 0.78, 0.82),
        (0.95, 0.40, 0.15),
        (0.98, 0.30, 0.10),
    ]
    return LinearSegmentedColormap.from_list("ow_aj1_white", colors, N=256)


def render(freqs, times, Sxx_db, anomaly_mask, annotations, sample_rate,
           output_path="rf_drone_output.png"):

    BG = "#f5f3ee"
    TEXT_DARK = "#1a1a1a"
    TEXT_MID = "#8a8680"
    CYAN = "#4dc8d1"
    ORANGE = "#f26522"
    BEIGE = "#d9d2c5"
    CREAM = "#f0ece4"

    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    gs = gridspec.GridSpec(
        5, 2,
        height_ratios=[0.06, 1.0, 0.25, 0.30, 0.05],
        width_ratios=[1, 0.03],
        hspace=0.18, wspace=0.03
    )

    cmap = create_colormap()
    freqs_mhz = freqs / 1e6

    # --- TITLE ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(BG)
    ax_title.axis("off")
    ax_title.text(0.0, 0.9, '"DRONE DETECT"', transform=ax_title.transAxes,
                  fontsize=34, fontweight="bold", fontfamily="monospace",
                  color=TEXT_DARK, va="top")
    ax_title.text(0.0, 0.2,
                  "RF WATERFALL  /  DRONE SIGNATURE ANALYSIS  /  FHSS  /  VIDEO DOWNLINK  /  TELEMETRY",
                  transform=ax_title.transAxes, fontsize=9,
                  fontfamily="monospace", color=TEXT_MID, va="top")
    ax_title.text(1.0, 0.9, "MAKING THE\nINVISIBLE VISIBLE\nIN THE SPECTRUM",
                  transform=ax_title.transAxes, fontsize=10,
                  fontfamily="monospace", color=ORANGE,
                  va="top", ha="right", linespacing=1.5)

    # --- MAIN WATERFALL ---
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.set_facecolor(CREAM)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 99)
    im = ax_main.pcolormesh(times, freqs_mhz, Sxx_db, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading="nearest",
                            rasterized=True)
    ax_main.contour(times, freqs_mhz, anomaly_mask.astype(float),
                    levels=[0.5], colors=[CYAN], linewidths=0.3, alpha=0.3)

    drone_anns = [a for a in annotations if a["type"] == "drone"]
    other_anns = [a for a in annotations if a["type"] != "drone"]

    for anom in drone_anns:
        t_start, t_end = anom["time"]
        freq_val = float(anom["freq"])
        ax_main.axvspan(t_start, t_end, alpha=0.04, color=ORANGE)
        t_mid = (t_start + t_end) / 2
        label = f'"{anom["label"]}"'
        y_off = 0.08 if freq_val >= 0 else -0.08
        ax_main.annotate(
            label, xy=(t_mid, freq_val),
            xytext=(t_mid + 0.15, freq_val + y_off),
            fontsize=8, fontfamily="monospace", fontweight="bold",
            color=TEXT_DARK,
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2),
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor=ORANGE, linewidth=1.2)
        )
        for te in [t_start, t_end]:
            ax_main.axvline(x=te, color=CYAN, linestyle=":", linewidth=0.5, alpha=0.4)

    for anom in other_anns:
        t_mid = (anom["time"][0] + anom["time"][1]) / 2
        freq_val = float(anom["freq"])
        ax_main.annotate(
            f'"{anom["label"]}"', xy=(t_mid, freq_val),
            xytext=(t_mid + 0.2, freq_val + 0.06),
            fontsize=8, fontfamily="monospace", fontweight="bold",
            color=TEXT_MID,
            arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.0),
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor=CYAN, linewidth=1.0)
        )

    ax_main.set_ylabel("FREQUENCY (MHz)", fontsize=10, fontfamily="monospace",
                       color=TEXT_DARK, labelpad=10)
    ax_main.set_xlabel("TIME (seconds)", fontsize=10, fontfamily="monospace",
                       color=TEXT_DARK, labelpad=10)
    ax_main.tick_params(colors=TEXT_MID, labelsize=8)
    for sp in ax_main.spines.values():
        sp.set_color(BEIGE)

    # --- COLORBAR ---
    ax_cb = fig.add_subplot(gs[1, 1])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label("POWER (dB)", fontsize=9, fontfamily="monospace",
                 color=TEXT_MID, labelpad=10)
    cb.ax.tick_params(colors=TEXT_MID, labelsize=7)
    cb.outline.set_edgecolor(BEIGE)

    # --- POWER vs TIME ---
    ax_power = fig.add_subplot(gs[2, 0])
    ax_power.set_facecolor(CREAM)
    mean_power = np.mean(Sxx_db, axis=0)
    max_power = np.max(Sxx_db, axis=0)
    ax_power.fill_between(times, mean_power, alpha=0.2, color=ORANGE)
    ax_power.plot(times, mean_power, color=ORANGE, linewidth=0.8, label="MEAN POWER")
    ax_power.plot(times, max_power, color=CYAN, linewidth=0.5, alpha=0.6, label="PEAK POWER")
    for anom in drone_anns:
        ax_power.axvspan(anom["time"][0], anom["time"][1], alpha=0.06, color=ORANGE)
    ax_power.set_ylabel("dB", fontsize=9, fontfamily="monospace", color=TEXT_DARK)
    ax_power.set_xlabel("TIME (seconds)", fontsize=9, fontfamily="monospace", color=TEXT_DARK)
    ax_power.tick_params(colors=TEXT_MID, labelsize=7)
    ax_power.legend(fontsize=7, loc="upper right", facecolor="white",
                    edgecolor=BEIGE, labelcolor=TEXT_DARK)
    for sp in ax_power.spines.values():
        sp.set_color(BEIGE)

    # --- DRONE TIMELINE (one row per signal) ---
    ax_tl = fig.add_subplot(gs[3, 0])
    ax_tl.set_facecolor(CREAM)

    # assign a color per drone ID
    drone_id_colors = {}
    color_palette = [ORANGE, CYAN, "#d4a574"]
    color_idx = 0

    row_labels = []
    for i, a in enumerate(drone_anns):
        parts = a["label"].split(" ")
        drone_id = parts[0] + " " + parts[1]
        sig_type = " ".join(parts[2:])

        if drone_id not in drone_id_colors:
            drone_id_colors[drone_id] = color_palette[color_idx % len(color_palette)]
            color_idx += 1

        color = drone_id_colors[drone_id]
        ts, te = a["time"]

        ax_tl.barh(i, te - ts, left=ts, height=0.7,
                   color=color, alpha=0.6, edgecolor=color, linewidth=0.8)

        # label aligned at fixed x position
        ax_tl.text(3.5, i, f'"{sig_type}"', fontsize=7,
                   fontfamily="monospace", fontweight="bold",
                   color=TEXT_DARK, ha="left", va="center")

        row_labels.append(f'"{drone_id}"')

    ax_tl.set_yticks(range(len(drone_anns)))
    ax_tl.set_yticklabels(row_labels, fontsize=7, fontfamily="monospace", color=TEXT_DARK)
    ax_tl.set_xlabel("TIME (seconds)", fontsize=9, fontfamily="monospace", color=TEXT_DARK)
    ax_tl.set_xlim(ax_main.get_xlim())
    ax_tl.tick_params(colors=TEXT_MID, labelsize=7)
    ax_tl.set_title('"DRONE ACTIVITY TIMELINE"', fontsize=10,
                    fontfamily="monospace", fontweight="bold",
                    color=TEXT_DARK, loc="left", pad=8)
    ax_tl.invert_yaxis()
    for sp in ax_tl.spines.values():
        sp.set_color(BEIGE)

    # --- FOOTER ---
    ax_ft = fig.add_subplot(gs[4, :])
    ax_ft.set_facecolor(BG)
    ax_ft.axis("off")
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ax_ft.text(0.0, 0.5,
               f"SAMPLE RATE: {sample_rate/1e6:.3f} MHz   |   "
               f"FFT: 512   |   "
               f"DRONES DETECTED: {len(drone_id_colors)}   |   "
               f"SIGNALS: {len(drone_anns)}   |   {timestamp}",
               transform=ax_ft.transAxes, fontsize=8,
               fontfamily="monospace", color=TEXT_MID, va="center")
    ax_ft.text(1.0, 0.8, '"EVERYTHING IN QUOTES"', transform=ax_ft.transAxes,
               fontsize=8, fontfamily="monospace", color=ORANGE,
               va="center", ha="right")
    ax_ft.text(1.0, 0.15, 'Sarom Leang, "PhD."', transform=ax_ft.transAxes,
               fontsize=9, fontfamily="monospace", color=TEXT_DARK,
               va="center", ha="right")

    # UNCLASSIFIED banners (top and bottom)
    fig.text(0.5, 0.995, "UNCLASSIFIED", fontsize=14, fontweight="bold",
             fontfamily="monospace", color="#00A600", ha="center", va="top")
    fig.text(0.5, 0.005, "UNCLASSIFIED", fontsize=14, fontweight="bold",
             fontfamily="monospace", color="#00A600", ha="center", va="bottom")

    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print('"DRONE DETECT"')
    print("Identifying drone signatures in the RF spectrum.\n")

    print("[1/4] Generating RF environment with drone signals...")
    iq, sr, anns = generate_drone_rf()
    print(f"      {len(iq):,} samples at {sr/1e6:.3f} MHz")

    print("[2/4] Computing waterfall spectrogram...")
    freqs, times, Sxx_db = compute_waterfall(iq, sr)
    print(f"      Shape: {Sxx_db.shape[0]} x {Sxx_db.shape[1]}")

    print("[3/4] Detecting anomalies...")
    anomaly_mask = detect_anomalies(Sxx_db)
    print(f"      {100*np.mean(anomaly_mask):.2f}% flagged")

    print("[4/4] Rendering...")
    render(freqs, times, Sxx_db, anomaly_mask, anns, sr)
    print('"COMPLETE"')


if __name__ == "__main__":
    main()
