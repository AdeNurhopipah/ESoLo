# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.io import loadmat
from tqdm import tqdm
from os import path
from scipy.stats import pearsonr

# ============================================================
# LOAD LEADFIELD
# ============================================================


def load_leadfield(lf_path):
    """Load Brainstorm leadfield."""

    if not os.path.isfile(lf_path):
        raise FileNotFoundError(f"Leadfield file not found:\n{lf_path}")

    mat = loadmat(
        lf_path,
        struct_as_record=False,
        squeeze_me=True,
    )

    if "lf" not in mat:
        raise KeyError(f"Variable 'lf' not found:\n{lf_path}")

    lf = mat["lf"]

    required_fields = ("pos", "leadfield", "chanlocs")
    missing_fields = [field for field in required_fields if not hasattr(lf, field)]

    if missing_fields:
        raise AttributeError(f"Leadfield fields not found: {missing_fields}")

    source_pos = np.asarray(
        lf.pos,
        dtype=np.float64,
    )

    leadfield = np.asarray(
        lf.leadfield,
        dtype=np.float64,
    )

    chanlocs = lf.chanlocs

    if source_pos.ndim != 2 or source_pos.shape[1] != 3:
        raise ValueError(
            "source_pos must have shape "
            f"(n_sources, 3), got {source_pos.shape}"
        )

    if leadfield.ndim != 3 or leadfield.shape[-1] != 3:
        raise ValueError(
            "leadfield must have shape "
            f"(n_channels, n_sources, 3), got {leadfield.shape}"
        )

    if source_pos.shape[0] != leadfield.shape[1]:
        raise ValueError(
            "The number of source positions does not match "
            "the number of leadfield sources."
        )

    return source_pos, leadfield, chanlocs


# ============================================================
# MAP STIMULATION TO LEADFIELD SOURCES
# ============================================================


def map_stim_to_sources(
    df_stim,
    source_pos,
    coordinate_columns=("stim_x", "stim_y", "stim_z"),
):
    """Map each stimulation coordinate to its nearest source."""

    df_mapped = df_stim.copy()

    source_pos = np.asarray(
        source_pos,
        dtype=np.float64,
    )

    missing_columns = [
        column for column in coordinate_columns if column not in df_mapped.columns
    ]

    if missing_columns:
        raise ValueError(f"Coordinate columns not found: {missing_columns}")

    stim_xyz = df_mapped[list(coordinate_columns)].to_numpy(dtype=np.float64)

    if not np.isfinite(stim_xyz).all():
        raise ValueError("Stimulation coordinates contain NaN or Inf values.")

    source_indices = []
    source_distances = []
    source_coordinates = []

    for xyz in stim_xyz:
        distances = np.linalg.norm(
            source_pos - xyz,
            axis=1,
        )

        source_idx = int(np.argmin(distances))

        source_indices.append(source_idx)
        source_distances.append(float(distances[source_idx]))
        source_coordinates.append(source_pos[source_idx])

    source_coordinates = np.asarray(
        source_coordinates,
        dtype=np.float64,
    )

    df_mapped["source_idx"] = source_indices
    df_mapped["source_x"] = source_coordinates[:, 0]
    df_mapped["source_y"] = source_coordinates[:, 1]
    df_mapped["source_z"] = source_coordinates[:, 2]
    df_mapped["source_distance_mm"] = source_distances

    df_source = (
        df_mapped.sort_values("source_distance_mm")
        .drop_duplicates(
            subset="source_idx",
            keep="first",
        )
        .reset_index(drop=True)
    )

    unique_source_indices = df_source["source_idx"].to_numpy(dtype=int)

    return (
        df_mapped,
        df_source,
        unique_source_indices,
    )


# ============================================================
# LOAD REAL EEG
# ============================================================


def load_eeg_files(file_paths):
    """Load and combine EEG data into (trial, channel, time) format."""

    data_all = []

    for file_path in file_paths:
        data = np.load(file_path)

        if data.ndim == 2:
            data = data[None, ...]

        if data.ndim != 3:
            raise ValueError(f"{file_path}: expected 2D/3D, got {data.shape}")

        data_all.append(data.astype(np.float32))

    if not data_all:
        raise ValueError("No EEG files could be loaded.")

    shapes = {data.shape[1:] for data in data_all}

    if len(shapes) != 1:
        raise ValueError(f"Channel/time shapes differ: {shapes}")

    return np.concatenate(
        data_all,
        axis=0,
    )


# ============================================================
# FARTHEST-POINT SAMPLING
# ============================================================


def farthest_point_sampling(
    positions,
    n_select,
    random_state=42,
):
    """Select spatially distributed positions using farthest-point sampling."""
    positions = np.asarray(positions, dtype=np.float64)

    if n_select > len(positions):
        raise ValueError(
            f"Requested {n_select} sources, " f"but only {len(positions)} available."
        )

    rng = np.random.default_rng(random_state)

    selected = np.empty(n_select, dtype=int)

    # First source is random
    first_idx = rng.integers(len(positions))
    selected[0] = first_idx

    # Distance to nearest selected source
    min_distances = np.linalg.norm(
        positions - positions[first_idx],
        axis=1,
    )

    min_distances[first_idx] = 0.0

    for i in range(1, n_select):
        next_idx = np.argmax(min_distances)
        selected[i] = next_idx

        new_distances = np.linalg.norm(
            positions - positions[next_idx],
            axis=1,
        )

        min_distances = np.minimum(
            min_distances,
            new_distances,
        )

        min_distances[next_idx] = 0.0

    return np.sort(selected)


# ============================================================
# COVARIANCE-BASED NOISE
# ============================================================


def compute_noise_covariance(
    real_data,
    baseline=(0, 200),
    shrinkage=0.05,
    regularization=1e-8,
    average_reference=True,
):
    """Estimate channel covariance from real EEG baseline."""
    data = np.asarray(real_data, dtype=np.float64)

    if data.ndim != 3:
        raise ValueError(
            "real_data must have shape (trial, channel, time), "
            f"got {data.shape}"
        )

    start, end = baseline
    if not 0 <= start < end <= data.shape[-1]:
        raise ValueError(
            f"Baseline {baseline} is invalid for {data.shape[-1]} time points."
        )

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be within the range [0, 1].")

    baseline_data = data[..., start:end].copy()
    baseline_data -= baseline_data.mean(axis=-1, keepdims=True)

    if average_reference:
        baseline_data -= baseline_data.mean(axis=1, keepdims=True)

    # Treat each time sample from every trial as one observation.
    observations = baseline_data.transpose(0, 2, 1).reshape(-1, baseline_data.shape[1])
    observations -= observations.mean(axis=0, keepdims=True)

    covariance = np.cov(observations, rowvar=False, bias=False)
    covariance = 0.5 * (covariance + covariance.T)

    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal

    mean_variance = float(np.mean(np.diag(covariance)))
    covariance += (regularization * max(mean_variance, 1e-12)) * np.eye(
        covariance.shape[0]
    )

    return covariance.astype(np.float64)


def generate_covariance_based_noise(
    covariance,
    n_generate,
    n_times,
    random_state=None,
    average_reference=True,
):
    """Generate spatially correlated Gaussian EEG noise.

    The generated samples have an empirical channel covariance that
    approaches the supplied covariance as the number of time samples grows.
    Output shape is (n_generate, n_channels, n_times).
    """
    covariance = np.asarray(covariance, dtype=np.float64)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "covariance must be a square matrix, "
            f"got {covariance.shape}"
        )

    if n_generate <= 0 or n_times <= 0:
        raise ValueError("n_generate and n_times must be greater than zero.")

    covariance = 0.5 * (covariance + covariance.T)

    # Eigen-decomposition is more robust than Cholesky for rank-deficient
    # average-referenced EEG covariance matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    max_eigenvalue = max(float(eigenvalues.max()), 1e-12)
    eigenvalues = np.clip(eigenvalues, max_eigenvalue * 1e-12, None)
    covariance_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    rng = np.random.default_rng(random_state)
    white_noise = rng.standard_normal(size=(n_generate, covariance.shape[0], n_times))

    noise = np.einsum("ij,bjt->bit", covariance_sqrt, white_noise, optimize=True)

    noise -= noise.mean(axis=-1, keepdims=True)

    if average_reference:
        noise -= noise.mean(axis=1, keepdims=True)

    return noise.astype(np.float32)


# ============================================================
# ERP WAVEFORM
# ============================================================


def gaussian_peak(
    time,
    peak_ms,
    width_ms,
    amplitude,
):
    """Generate one Gaussian ERP component."""

    return amplitude * np.exp(-0.5 * ((time * 1000 - peak_ms) / width_ms) ** 2)


# ============================================================
# FREE-ORIENTATION LEADFIELD
# ============================================================


def get_oriented_leadfield(
    leadfield,
    source_idx,
    rng,
):
    """Project a free-orientation leadfield onto a random unit orientation."""

    if leadfield.ndim != 3 or leadfield.shape[-1] != 3:
        raise ValueError(
            "leadfield must have shape "
            f"(n_channels, n_sources, 3), got {leadfield.shape}"
        )

    if not 0 <= source_idx < leadfield.shape[1]:
        raise IndexError(f"source_idx must be between 0 and {leadfield.shape[1] - 1}")

    orientation = rng.normal(size=3)

    orientation /= np.linalg.norm(orientation) + 1e-12

    oriented_leadfield = leadfield[:, source_idx, :] @ orientation

    return (
        oriented_leadfield.astype(np.float32),
        orientation.astype(np.float32),
    )


# ============================================================
# GENERATE ERP EPOCHS
# ============================================================
def generate_erp_epochs(
    source_pos,
    leadfield,
    source_idx,
    n_trials=30,
    n_times=1000,
    sfreq=1000,
    n1=(300, 35, -1.0),
    p1=(25, 0.5),
    n2=(60, -0.8),
    p2=(45, 0.4),
    p1_delay=(60, 10),
    n2_delay=(90, 15),
    p2_delay=(100, 20),
    n1_time_sd=10,
    width_sd=5,
    amplitude_sd=0.1,
    average_reference=True,
    random_state=None,
):
    """Generate ERP epochs from a single active source."""

    n_channels, n_sources = leadfield.shape[:2]

    if not 0 <= source_idx < n_sources:
        raise IndexError(f"source_idx must be between 0 and {n_sources - 1}")

    rng = np.random.default_rng(random_state)

    time = np.arange(n_times) / sfreq

    epochs = np.empty(
        (n_trials, n_channels, n_times),
        dtype=np.float32,
    )

    waveforms = np.empty(
        (n_trials, n_times),
        dtype=np.float32,
    )

    trial_metadata = []

    for trial in range(n_trials):
        oriented_lf, orientation = get_oriented_leadfield(
            leadfield,
            source_idx,
            rng,
        )

        n1_time = rng.normal(
            n1[0],
            n1_time_sd,
        )

        p1_time = n1_time + max(rng.normal(*p1_delay), 1)

        n2_time = p1_time + max(rng.normal(*n2_delay), 1)

        p2_time = n2_time + max(rng.normal(*p2_delay), 1)

        components = [
            ("N1", n1_time, n1[1], n1[2]),
            ("P1", p1_time, p1[0], p1[1]),
            ("N2", n2_time, n2[0], n2[1]),
            ("P2", p2_time, p2[0], p2[1]),
        ]

        waveform = np.zeros(
            n_times,
            dtype=np.float32,
        )

        component_metadata = {}

        for name, peak_time, width, amplitude in components:
            trial_width = max(
                rng.normal(width, width_sd),
                1,
            )

            trial_amplitude = rng.normal(
                amplitude,
                abs(amplitude) * amplitude_sd,
            )

            waveform += gaussian_peak(
                time,
                peak_time,
                trial_width,
                trial_amplitude,
            ).astype(np.float32)

            name = name.lower()

            component_metadata[f"{name}_time_ms"] = peak_time

            component_metadata[f"{name}_width_ms"] = trial_width

            component_metadata[f"{name}_amplitude"] = trial_amplitude

        erp = oriented_lf[:, None] * waveform[None, :]

        if average_reference:
            erp -= erp.mean(
                axis=0,
                keepdims=True,
            )

        epochs[trial] = erp
        waveforms[trial] = waveform

        trial_metadata.append(
            {
                "trial": trial + 1,
                "source_idx": int(source_idx),
                "source_x": source_pos[source_idx, 0],
                "source_y": source_pos[source_idx, 1],
                "source_z": source_pos[source_idx, 2],
                "orientation_x": orientation[0],
                "orientation_y": orientation[1],
                "orientation_z": orientation[2],
                **component_metadata,
            }
        )

    return (
        epochs,
        waveforms,
        pd.DataFrame(trial_metadata),
    )


# ============================================================
# GENERATE ERP DATASET
# ============================================================


def generate_erp_dataset(
    lf_path,
    out_dir,
    source_indices,
    noise_data,
    n_trials_per_source=30,
    n_times=1000,
    sfreq=1000,
    erp_window=(250, 650),
    snr_db_range=(-5.0, 10.0),
    random_state=42,
    overwrite=False,
):
    """
    Generate synthetic ERP dataset.
    """

    metadata_file = os.path.join(
        out_dir,
        "metadata.xlsx",
    )

    if os.path.isfile(metadata_file) and not overwrite:
        raise FileExistsError(f"Dataset already exists:\n{metadata_file}")

    eeg_dir = os.path.join(
        out_dir,
        "eeg",
    )

    waveform_dir = os.path.join(
        out_dir,
        "waveform",
    )

    os.makedirs(eeg_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)

    rng = np.random.default_rng(random_state)

    source_pos, leadfield, _ = load_leadfield(lf_path)

    noise_data = np.asarray(
        noise_data,
        dtype=np.float32,
    )

    if noise_data.ndim != 3:
        raise ValueError(f"noise_data must be 3D, got {noise_data.shape}")

    if noise_data.shape[1] != leadfield.shape[0]:
        raise ValueError("The noise and leadfield channel counts differ.")

    if noise_data.shape[-1] < n_times:
        raise ValueError("Noise has fewer time points than n_times.")

    noise_data = noise_data[
        ...,
        :n_times,
    ].copy()

    noise_data -= noise_data.mean(
        axis=-1,
        keepdims=True,
    )

    noise_data -= noise_data.mean(
        axis=1,
        keepdims=True,
    )

    source_indices = np.asarray(
        source_indices,
        dtype=int,
    )

    if np.any(source_indices < 0) or np.any(source_indices >= leadfield.shape[1]):
        raise IndexError("A source_idx is outside the leadfield bounds.")

    start, end = erp_window

    if not 0 <= start < end <= n_times:
        raise ValueError(f"Invalid ERP window: {erp_window}")

    rows = []

    # ========================================================
    # GENERATE SOURCE BY SOURCE
    # ========================================================

    for source_order, source_idx in enumerate(
        tqdm(
            source_indices,
            desc="Generate sources",
        )
    ):
        source_seed = int(rng.integers(0, 2**32 - 1))

        erp_epochs, waveforms, erp_metadata = generate_erp_epochs(
            source_pos=source_pos,
            leadfield=leadfield,
            source_idx=int(source_idx),
            n_trials=n_trials_per_source,
            n_times=n_times,
            sfreq=sfreq,
            random_state=source_seed,
        )

        # ----------------------------------------------------
        # Allocate arrays for all trials of this source
        # ----------------------------------------------------

        source_eeg = np.empty(
            (
                n_trials_per_source,
                leadfield.shape[0],
                n_times,
            ),
            dtype=np.float32,
        )

        source_waveforms = np.empty(
            (
                n_trials_per_source,
                n_times,
            ),
            dtype=np.float32,
        )

        source_name = f"source_{source_order:04d}" f"_idx_{int(source_idx):05d}"

        eeg_name = f"{source_name}.npy"
        waveform_name = f"{source_name}.npy"

        # ----------------------------------------------------
        # GENERATE TRIALS
        # ----------------------------------------------------

        for trial_idx in range(n_trials_per_source):
            noise_idx = int(rng.integers(len(noise_data)))
            noise = noise_data[noise_idx].copy()
            erp = erp_epochs[trial_idx]
            waveform = waveforms[trial_idx]

            # ------------------------------------------------
            # SNR SCALING
            # ------------------------------------------------

            noise_rms = np.sqrt(np.mean(noise[:, start:end] ** 2))
            erp_rms = np.sqrt(np.mean(erp[:, start:end] ** 2))
            snr_db = float(rng.uniform(*snr_db_range))
            target_erp_rms = noise_rms * 10 ** (snr_db / 20)
            erp_scale = target_erp_rms / (erp_rms + 1e-12)
            scaled_erp = (erp * erp_scale).astype(np.float32)
            scaled_waveform = (waveform * erp_scale).astype(np.float32)

            # ------------------------------------------------
            # SYNTHETIC EEG
            # ------------------------------------------------

            synthetic = (noise + scaled_erp).astype(np.float32)
            synthetic -= synthetic.mean(
                axis=-1,  keepdims=True,
            )
            synthetic -= synthetic.mean(
                axis=0,  keepdims=True,
            )

            # ------------------------------------------------
            # STORE IN SOURCE ARRAY
            # ------------------------------------------------

            source_eeg[trial_idx] = synthetic
            source_waveforms[trial_idx] = scaled_waveform

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            row = {
                "source_order": source_order,
                "source_idx": int(source_idx),
                "trial_idx": trial_idx,
                "noise_idx": noise_idx,
                "snr_db": snr_db,
                "erp_scale": float(erp_scale),
                "noise_rms": float(noise_rms),
                "erp_rms_original": float(erp_rms),
                "erp_rms_scaled": float(
                    np.sqrt(np.mean(scaled_erp[:, start:end] ** 2))
                ),
                "synthetic_std": float(synthetic.std()),
                "synthetic_rms": float(np.sqrt(np.mean(synthetic**2))),
                "synthetic_ptp": float(np.ptp(synthetic)),
                "eeg_file": os.path.join(
                    "eeg", eeg_name,),
                "waveform_file": os.path.join(
                    "waveform", waveform_name,),
            }

            row.update(erp_metadata.iloc[trial_idx].to_dict())

            rows.append(row)

        # ----------------------------------------------------
        # SAVE ONE FILE PER SOURCE
        # ----------------------------------------------------

        np.save(
            os.path.join(
                eeg_dir, eeg_name,
            ), source_eeg,
        )

        np.save(
            os.path.join(
                waveform_dir, waveform_name,
            ), source_waveforms,
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = pd.DataFrame(rows)

    metadata.to_excel(
        metadata_file, index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("Dataset generated")
    print("-----------------")
    print("Output        :", out_dir)
    print("Sources       :", len(source_indices))
    print("Trials/source :", n_trials_per_source)
    print("EEG files     :", len(source_indices))
    print("Waveform files:", len(source_indices))
    print("Total trials  :", len(metadata))

    return metadata


# ============================================================
# INDEPENDENT GAUSSIAN NOISE
# ============================================================
def generate_independent_gaussian_noise(
    n_generate,
    n_channels,
    n_times,
    random_state=42,
    average_reference=True,
    remove_channel_mean=True,
):
    """Generate independent white Gaussian noise.

    Returns
    -------
    noise : np.ndarray
        Shape (n_generate, n_channels, n_times).
    """
    rng = np.random.default_rng(random_state)

    noise = rng.standard_normal(size=(n_generate, n_channels, n_times)).astype(
        np.float32
    )

    if remove_channel_mean:
        noise -= noise.mean(axis=2, keepdims=True)

    if average_reference:
        noise -= noise.mean(axis=1, keepdims=True)

    return noise


# ============================================================
# Helpers Similarity and Distance Metrics
# ============================================================
def load_npy_folder(folder):
    """Load and concatenate all NumPy arrays in a folder."""
    files = sorted(
        path.join(folder, file) for file in os.listdir(folder) if file.endswith(".npy")
    )
    if not files:
        raise FileNotFoundError(f"No .npy files: {folder}")

    arrays = [np.load(file).astype(np.float64) for file in files]
    return np.concatenate(
        [array if array.ndim == 3 else array[None] for array in arrays],
        axis=0,
    )


def get_baseline(data, baseline=(0, 200), sfreq=1000):
    """Extract the requested baseline interval from EEG data."""
    start = int(baseline[0] * sfreq / 1000)
    stop = int(baseline[1] * sfreq / 1000)
    return data[..., start:stop]


def spatial_covariance(data):
    """Calculate channel covariance across trials and time points."""
    # trials × channels × time → channels × observations
    values = data.transpose(1, 0, 2)
    values = values.reshape(values.shape[0], -1)
    values -= values.mean(axis=1, keepdims=True)
    return np.cov(values)


def spatial_correlation(data):
    """Calculate channel correlation across trials and time points."""
    values = data.transpose(1, 0, 2)
    values = values.reshape(values.shape[0], -1)
    return np.corrcoef(values)


def normalized_covariance(covariance):
    """Normalize a covariance matrix by its trace."""
    trace = np.trace(covariance)
    if trace <= 0:
        raise ValueError("Covariance trace must be positive.")
    return covariance / trace


def covariance_distance(real_cov, synthetic_cov):
    """Calculate Frobenius distance between normalized covariance matrices."""
    real_cov = normalized_covariance(real_cov)
    synthetic_cov = normalized_covariance(synthetic_cov)
    return np.linalg.norm(real_cov - synthetic_cov, ord="fro")


def matrix_correlation(real_corr, synthetic_corr):
    """Correlate the upper triangles of two correlation matrices."""
    upper_triangle = np.triu_indices_from(real_corr, k=1)
    return pearsonr(real_corr[upper_triangle], synthetic_corr[upper_triangle])


def significance_marker(p_value):
    """Convert a p-value into its conventional significance marker."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"
