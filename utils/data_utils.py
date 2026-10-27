import os
import re

from itertools import combinations
from os import path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Basic analysis helpers
# =============================================================================
def band_power(psds, freqs, lo, hi):
    idx = (freqs >= lo) & (freqs <= hi)
    return psds[:, idx].mean()


def get_evoked_peak(data, info, n_epoch):
    evoked_data = data.mean(axis=0)
    evoked = mne.EvokedArray(evoked_data, info, tmin=0, nave=n_epoch)

    peak_ch, peak_time = evoked.get_peak(
        ch_type="eeg",
        tmin=0,
        tmax=0.99,
        mode="abs",
    )

    peak_idx = np.argmin(np.abs(evoked.times - peak_time))
    peak_values = evoked.data[:, peak_idx]
    peak_amp_abs_uv = np.abs(
        evoked.data[evoked.ch_names.index(peak_ch), peak_idx]
    )

    return evoked, peak_ch, peak_time, peak_values, peak_amp_abs_uv


def get_evoked_psd(evoked_data, n_times, fmin=0, fmax=100):
    psd = evoked_data.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=min(512, n_times),
        verbose=0,
    )

    return psd.get_data(), psd.freqs


def get_correlation(data, n_ch, evoked_data=None):
    # Correlation from all epochs: channel x (epoch × time)
    trans_data = data.transpose(1, 0, 2).reshape(n_ch, -1)
    ch_data_corr = np.corrcoef(trans_data)

    # Correlation from evoked: channel x time
    ch_evoked_corr = None

    if evoked_data is not None:
        evoked_array = evoked_data.data if hasattr(evoked_data, "data") else evoked_data
        ch_evoked_corr = np.corrcoef(evoked_array)

    return ch_data_corr, ch_evoked_corr


# =============================================================================
# Data exploration
# =============================================================================
def summarize_evoked(
    data,
    data_file,
    subject,
    stim_channel,
    stim_intensity,
    info,
    n_epoch,
    n_ch,
    n_times,
    in_v=False,
    n_top_elec=5,
    nearest_elec=None,
    nearest_dist=None,
):
    if in_v:
        data = data * 1e6

    evoked, peak_ch, peak_time, peak_values, peak_amp_abs_uv = get_evoked_peak(
        data, info, n_epoch
    )

    epoch_std = data.std(axis=(1, 2))
    psds, freqs = get_evoked_psd(evoked, n_times)

    ch_data_corr, _ = get_correlation(data, n_ch)
    corr_offdiag = ch_data_corr[
        ~np.eye(ch_data_corr.shape[0], dtype=bool)
    ]

    top_idx = np.argsort(np.abs(peak_values))[-n_top_elec:][::-1]
    top_electrodes = [evoked.ch_names[j] for j in top_idx]

    if data_file == "all_run":
        run = "all_run"
    else:
        r = re.search(r"(run-\d+)", data_file)
        run = r.group(1)

    stats = {
        "subject": subject,
        "run": run,
        "stim_channel": stim_channel,
        "stim_intensity": stim_intensity,
        "n_epoch": n_epoch,

        "evoked_peak_channel": peak_ch,
        "evoked_peak_time_ms": peak_time * 1000,
        "evoked_peak_amp_abs_uV": peak_amp_abs_uv,
        "epoch_std_avg": epoch_std.mean(),

        "psd_mean": psds.mean(),
        "psd_delta_1_4": band_power(psds, freqs, 1, 4),
        "psd_theta_4_8": band_power(psds, freqs, 4, 8),
        "psd_alpha_8_13": band_power(psds, freqs, 8, 13),
        "psd_beta_13_30": band_power(psds, freqs, 13, 30),
        "psd_gamma_30_80": band_power(psds, freqs, 30, 80),
        "corr_abs_avg": np.abs(corr_offdiag).mean(),

        "top_peak_elec": top_electrodes,
        "nearest_elec": nearest_elec,
        "nearest_dist": nearest_dist,
    }

    return stats


# =============================================================================
# Subject information
# =============================================================================
def get_subject_info(data_path, subject):
    data_folder = path.join(data_path, subject, "eeg")

    ch_path = [
        f for f in os.listdir(data_folder)
        if f.endswith("electrodes.tsv")
    ]

    ch_link = path.join(data_folder, ch_path[0])

    electrodes = pd.read_csv(ch_link, sep="\t")
    ch_names = electrodes["name"].astype(str).tolist()
    coords = electrodes[["x", "y", "z"]].to_numpy(dtype=float) / 1000.0
    ch_pos = dict(zip(ch_names, coords))

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head",
    )

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=1000,
        ch_types="eeg",
    )

    info.set_montage(montage, on_missing="ignore")

    return info


# =============================================================================
# Data loading helpers
# =============================================================================
def get_run_from_filename(filename):
    r = re.search(r"(run-\d+)", filename)
    return r.group(1) if r else None


def get_metadata(
    df_stats,
    df_stim,
    subject,
    run=None,
    stim_channel=None,
):
    if run is not None:
        match_stats = df_stats.loc[
            (df_stats["subject"] == subject)
            & (df_stats["run"] == run)
        ]

        if match_stats.empty:
            return None

        row_stats = match_stats.iloc[0]
        stim_channel = row_stats["stim_channel"]
        intensity = row_stats["stim_intensity"]

    else:
        match_stats = df_stats.loc[
            (df_stats["subject"] == subject)
            & (df_stats["stim_channel"] == stim_channel)
        ]

        intensity = (
            match_stats["stim_intensity"].iloc[0]
            if not match_stats.empty
            else np.nan
        )

    match_stim = df_stim.loc[
        (df_stim["subject"] == subject)
        & (df_stim["stim_channel"] == stim_channel)
    ]

    if match_stim.empty:
        stim_x = stim_y = stim_z = np.nan
        nyh_region = None
        nyh_distance_mm = np.nan

    else:
        row_stim = match_stim.iloc[0]

        stim_x = row_stim["stim_x"]
        stim_y = row_stim["stim_y"]
        stim_z = row_stim["stim_z"]
        nyh_region = row_stim["nyh_region"]
        nyh_distance_mm = row_stim["nyh_distance_mm"]

    return {
        "subject": subject,
        "run": run,
        "stim_channel": stim_channel,
        "stim_intensity": intensity,
        "stim_coord": [
            f"{stim_x:.1f}",
            f"{stim_y:.1f}",
            f"{stim_z:.1f}",
        ],
        "nyh_region": nyh_region,
        "nyh_distance_mm": nyh_distance_mm,
    }


def analyse_data(data, info, n_top_elec):
    n_epoch, n_ch, n_times = data.shape
    epoch_std = data.std(axis=(1, 2))

    evoked, peak_ch, peak_time, peak_values, peak_amp_abs_uv = get_evoked_peak(
        data, info, n_epoch
    )

    psds, freqs = get_evoked_psd(evoked, n_times)
    ch_data_corr, ch_evoked_corr = get_correlation(data, n_ch, evoked)

    corr_offdiag = ch_data_corr[
        ~np.eye(ch_data_corr.shape[0], dtype=bool)
    ]

    top_idx = np.argsort(np.abs(peak_values))[-n_top_elec:][::-1]
    top_electrodes = [evoked.ch_names[i] for i in top_idx]

    return {
        "evoked": evoked,
        "epoch_std": epoch_std,
        "peak_ch": peak_ch,
        "peak_time": peak_time,
        "peak_values": peak_values,
        "peak_amp_abs_uv": peak_amp_abs_uv,
        "psds": psds,
        "freqs": freqs,
        "ch_data_corr": ch_data_corr,
        "ch_evoked_corr": ch_evoked_corr,
        "corr_offdiag": corr_offdiag,
        "top_idx": top_idx,
        "top_electrodes": top_electrodes,
    }


def load_data_from_subject(
    data_path,
    subject,
    run,
    mode="real",
):
    if mode == "real":
        folder_link = path.join(data_path, subject, "eeg")
        files = os.listdir(folder_link)

        file = next(
            (
                f for f in files
                if subject in f and run in f and f.endswith(".npy")
            ),
            None,
        )

        if file is None:
            raise FileNotFoundError(f"No file found for {subject} {run}")

        data = np.load(path.join(folder_link, file))
        data = data[:, :, :1000] * 1e6

    elif mode == "syn":
        folder_link = path.join(data_path, subject)
        files = os.listdir(folder_link)

        file = next(
            (f for f in files if subject in f and run in f),
            None,
        )

        if file is None:
            raise FileNotFoundError(f"No file found for {subject} {run}")

        mat = loadmat(
            path.join(folder_link, file),
            squeeze_me=True,
            struct_as_record=False,
        )

        data = mat["data"]
        data = data.transpose(2, 0, 1)
        data = data - data.mean(axis=1, keepdims=True)

    return data


def extract_run(
    data_path,
    files,
    subject,
    info,
    df_stats,
    df_stim,
    n_top_elec=10,
    data_type="real",
):
    result_subject = []

    for f in files:
        run = get_run_from_filename(f)

        if run is None:
            continue

        meta = get_metadata(
            df_stats,
            df_stim,
            subject,
            run=run,
        )

        data = load_data_from_subject(
            data_path,
            subject,
            run,
            mode=data_type,
        )

        result = analyse_data(data, info, n_top_elec)

        label = (
            f"{run} | {subject}\n"
            f"{meta['stim_channel']} | {meta['stim_intensity']} mA\n"
            f"XYZ = {meta['stim_coord']}"
        )

        result_subject.append({
            "subject": subject,
            "run": run,
            "stim_channel": meta["stim_channel"],
            "stim_intensity": meta["stim_intensity"],
            "stim_coord": meta["stim_coord"],
            "nyh_region": meta["nyh_region"],
            "result": result,
            "label": label,
        })

    return result_subject


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_top_channels(ax, evoked, peak_time, top_idx):
    for ch in top_idx:
        ax.plot(
            evoked.times * 1000,
            evoked.data[ch],
            label=evoked.ch_names[ch],
        )

    ax.axvline(peak_time * 1000, ls="--", color="k")
    ax.set_title("Top 5 channels at global peak")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(fontsize=8)
    ax.grid(True)


def plot_gfp(ax, evoked, peak_time, peak_ch):
    times_ms = evoked.times * 1000
    gfp = np.std(evoked.data, axis=0)

    idx = np.argmax(gfp)
    gfp_time = times_ms[idx]

    ax.plot(times_ms, gfp)
    ax.axvline(gfp_time, ls="--", color="tab:blue")
    ax.axvline(peak_time * 1000, ls="--", color="k")

    ax.set_title(
        f"GFP Peak={gfp_time:.1f} ms\n"
        f"Global={peak_time * 1000:.1f} ms ({peak_ch})"
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("GFP (µV)")
    ax.grid(True)


def plot_topomap(
    ax,
    evoked,
    peak_values,
    peak_time,
    top_idx,
    top_elec,
):
    names = [""] * len(evoked.ch_names)

    for i in top_idx:
        names[i] = evoked.ch_names[i]

    mask = np.zeros(len(evoked.ch_names), dtype=bool)
    mask[top_idx] = True

    mne.viz.plot_topomap(
        peak_values,
        evoked.info,
        axes=ax,
        show=False,
        contours=6,
        names=names,
        mask=mask,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=10,
        ),
    )

    ax.set_title(
        f"Topomap @ {peak_time * 1000:.1f} ms\n{top_elec}",
        fontsize=9,
    )


def plot_result_grid(items, title, max_per_row=5):
    n_item = len(items)

    if n_item == 0:
        print(f"No items to plot: {title}")
        return

    n_col = min(max_per_row, n_item)
    n_block = int(np.ceil(n_item / max_per_row))
    n_row = n_block * 3

    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(5 * n_col, 4.5 * n_row),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for idx, item in enumerate(items):
        block = idx // max_per_row
        col = idx % max_per_row
        r0 = block * 3

        ax_top = axes[r0, col]
        ax_gfp = axes[r0 + 1, col]
        ax_map = axes[r0 + 2, col]

        for ax in [ax_top, ax_gfp, ax_map]:
            ax.axis("on")

        result = item["result"]
        top_idx = result["top_idx"]
        top_electrodes = [
            result["evoked"].ch_names[i]
            for i in top_idx
        ]

        plot_top_channels(
            ax_top,
            result["evoked"],
            result["peak_time"],
            result["top_idx"],
        )
        ax_top.set_title(item["label"], fontsize=9)

        plot_gfp(
            ax_gfp,
            result["evoked"],
            result["peak_time"],
            result["peak_ch"],
        )
        ax_gfp.set_title("GFP", fontsize=9)

        plot_topomap(
            ax_map,
            result["evoked"],
            result["peak_values"],
            result["peak_time"],
            result["top_idx"],
            top_electrodes,
        )
        ax_map.set_title("Topomap", fontsize=9)

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# =============================================================================
# Similarity helpers
# =============================================================================
def safe_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return pearsonr(x, y)[0]


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return spearmanr(x, y)[0]


def safe_cosine(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return np.nan

    return cosine_similarity(
        x.reshape(1, -1),
        y.reshape(1, -1),
    )[0, 0]


def elec_overlap(elec_a, elec_b):
    common = sorted(set(elec_a) & set(elec_b))
    union = set(elec_a) | set(elec_b)

    jaccard = len(common) / len(union)

    return common, jaccard


def top_electrode(
    result_a,
    result_b,
    n_top_elec=10,
    name_a="1",
    name_b="2",
):
    ch_names = result_a["evoked"].ch_names

    strength_a = np.max(np.abs(result_a["evoked"].data), axis=1)
    strength_b = np.max(np.abs(result_b["evoked"].data), axis=1)

    top_a = np.argsort(strength_a)[-n_top_elec:][::-1]
    top_b = np.argsort(strength_b)[-n_top_elec:][::-1]

    elec_a = [ch_names[i] for i in top_a]
    elec_b = [ch_names[i] for i in top_b]

    common, jaccard = elec_overlap(elec_a, elec_b)

    return {
        f"top{n_top_elec}_elec_{name_a}": elec_a,
        f"top{n_top_elec}_elec_{name_b}": elec_b,
        "common_elec": common,
        "jacard_elec": jaccard,
    }


def get_spatial_similarity(result_a, result_b):
    evoked_a = result_a["evoked"].data
    evoked_b = result_b["evoked"].data

    strength_a = np.sqrt(np.mean(evoked_a ** 2, axis=1))
    strength_b = np.sqrt(np.mean(evoked_b ** 2, axis=1))

    spatial_pearson = safe_pearson(strength_a, strength_b)
    spatial_cosine = safe_cosine(strength_a, strength_b)
    spatial_spearman = safe_spearman(strength_a, strength_b)

    spatial_sim_mean = (
        spatial_cosine
        + spatial_pearson
        + spatial_spearman
    ) / 3

    return {
        "spatial_pearson": spatial_pearson,
        "spatial_cosine": spatial_cosine,
        "spatial_spearman": spatial_spearman,
        "spatial_sim_mean": spatial_sim_mean,
    }


def get_frequency_similarity(result_a, result_b):
    psd_a = result_a["psds"]
    psd_b = result_b["psds"]

    freq_pearson = np.nanmean([
        safe_pearson(psd_a[ch], psd_b[ch])
        for ch in range(psd_a.shape[0])
    ])

    freq_cosine = np.nanmean([
        safe_cosine(psd_a[ch], psd_b[ch])
        for ch in range(psd_a.shape[0])
    ])

    freq_spearman = np.nanmean([
        safe_spearman(psd_a[ch], psd_b[ch])
        for ch in range(psd_a.shape[0])
    ])

    freq_sim_mean = (
        freq_pearson
        + freq_cosine
        + freq_spearman
    ) / 3

    return {
        "freq_pearson": freq_pearson,
        "freq_cosine": freq_cosine,
        "freq_spearman": freq_spearman,
        "freq_sim_mean": freq_sim_mean,
    }


def pairwise_similarity(
    results,
    group_label,
    n_top_elec,
    freq_mode="psd",
    name_a=None,
    name_b=None,
    skip_same_run=True,
):
    rows = []

    for a, b in combinations(results, 2):

        if skip_same_run:
            if a["subject"] == b["subject"] and a["run"] == b["run"]:
                continue

        spatial_sim = get_spatial_similarity(
            a["result"],
            b["result"],
        )

        overlap = top_electrode(
            a["result"],
            b["result"],
            n_top_elec=n_top_elec,
            name_a=name_a,
            name_b=name_b,
        )

        freq_sim = get_frequency_similarity(
            a["result"],
            b["result"],
        )

        rows.append({
            "group": group_label,

            f"subject_{name_a}": a["subject"],
            f"run_{name_a}": a["run"],
            f"stim_{name_a}": a["stim_channel"],
            f"coord_{name_a}": a["stim_coord"],

            f"subject_{name_b}": b["subject"],
            f"run_{name_b}": b["run"],
            f"stim_{name_b}": b["stim_channel"],
            f"coord_{name_b}": b["stim_coord"],

            f"stim_intensity_{name_a}": a["stim_intensity"],
            f"stim_intensity_{name_b}": b["stim_intensity"],

            **spatial_sim,
            **overlap,
            **freq_sim,
        })

    return pd.DataFrame(rows)