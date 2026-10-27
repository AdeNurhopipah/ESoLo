# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import argparse
from os import path

import numpy as np

project_path = path.dirname(path.dirname(path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.syn_gen_helpers import *


# ============================================================
# ARGUMENTS
# ============================================================

SUBJECTS_CHOICE =[ "sub-16", "sub-21", "sub-23", "sub-27",  "sub-33"]

parser = argparse.ArgumentParser()
parser.add_argument("--subject", default="sub-16", choices=SUBJECTS_CHOICE)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

gen_subject = args.subject
debug = args.debug

# ============================================================
# CONFIGURATION
# ============================================================

lf_mode = "vol"

sfreq = 1000
n_times = 1000

n_training_sources = 10 if debug else 300
n_trials_per_source = 3 if debug else 20
n_generations = 2 if debug else 10

n_noise_samples = 50 if debug else 300

baseline_window = (0, 200)
scaling_window = (250, 650)
snr_db_range = (-5.0, 10.0)

covariance_shrinkage = 0.05
covariance_regularization = 1e-8

random_state = 42
overwrite_dataset = True


# ============================================================
# PATHS
# ============================================================

data_root = path.join(project_path, "data")

real_folder = path.join(
    data_root,
    "ccep_coreg",
    "derivatives",
    "epochs",
    gen_subject,
    "eeg",
)

leadfield_file = path.join(
    data_root,
    "ccep_coreg_lf",
    f"{gen_subject}_lf_{lf_mode}.mat",
)

dataset_root = path.join(
    data_root,
    "syn_cov",
    lf_mode,
    gen_subject,
    *(["debug"] if debug else []),
)

os.makedirs(dataset_root, exist_ok=True)

if not path.isfile(leadfield_file):
    raise FileNotFoundError(leadfield_file)

if not path.isdir(real_folder):
    raise FileNotFoundError(real_folder)


# ============================================================
# LOAD LEADFIELD
# ============================================================

source_pos, leadfield, chanlocs = load_leadfield(
    leadfield_file
)

if source_pos.ndim != 2 or source_pos.shape[1] != 3:
    raise ValueError(
        f"Invalid source_pos shape: {source_pos.shape}"
    )

if leadfield.ndim != 3:
    raise ValueError(
        f"Invalid leadfield shape: {leadfield.shape}"
    )

n_channels, n_sources, n_orientations = leadfield.shape

if len(source_pos) != n_sources:
    raise ValueError(
        f"source_pos ({len(source_pos)}) != "
        f"leadfield sources ({n_sources})"
    )

print("\nLeadfield")
print("-" * 60)
print("Shape        :", leadfield.shape)
print("Channels     :", n_channels)
print("Sources      :", n_sources)
print("Orientations :", n_orientations)


# ============================================================
# DISTRIBUTED TRAINING SOURCES
# ============================================================

source_indices = farthest_point_sampling(
    positions=source_pos,
    n_select=n_training_sources,
    random_state=random_state,
)

selected_source_pos = source_pos[source_indices]

print("\nTraining sources")
print("-" * 60)
print("Method  : farthest-point sampling")
print("Selected:", len(source_indices))
print("Seed    :", random_state)


# ============================================================
# LOAD REAL EEG
# ============================================================

real_file_paths = sorted(
    path.join(real_folder, f)
    for f in os.listdir(real_folder)
    if f.endswith(".npy")
)

if not real_file_paths:
    raise FileNotFoundError(
        f"No .npy EEG files found in {real_folder}"
    )

real_data = load_eeg_files(real_file_paths)

if real_data.shape[-1] < n_times:
    raise ValueError(
        f"EEG has {real_data.shape[-1]} time points; "
        f"expected >= {n_times}."
    )

if real_data.shape[1] != n_channels:
    raise ValueError(
        f"EEG channels ({real_data.shape[1]}) != "
        f"leadfield channels ({n_channels})"
    )

real_data = real_data[..., :n_times].astype(
    np.float32
)

print("\nReal EEG")
print("-" * 60)
print("Shape:", real_data.shape)


# ============================================================
# NOISE COVARIANCE
# ============================================================

print("\nCompute noise covariance")
print("-" * 60)

noise_covariance = compute_noise_covariance(
    real_data=real_data,
    baseline=baseline_window,
    shrinkage=covariance_shrinkage,
    regularization=covariance_regularization,
    average_reference=True,
)

if noise_covariance.shape != (
    n_channels,
    n_channels,
):
    raise ValueError(
        f"Expected covariance "
        f"{(n_channels, n_channels)}, "
        f"got {noise_covariance.shape}"
    )

print("Covariance:", noise_covariance.shape)


# ============================================================
# GENERATE DATASET
# ============================================================

expected_samples = (
    len(source_indices)
    * n_trials_per_source
)

for gen_idx in range(1, n_generations + 1):

    gen_name = f"gen_{gen_idx}"
    generation_seed = random_state + gen_idx - 1

    output_dir = path.join(
        dataset_root,
        gen_name,
    )

    print(f"\n{'=' * 60}")
    print(
        f"Generate {gen_name} | "
        f"seed={generation_seed}"
    )
    print(f"{'=' * 60}")

    # --------------------------------------------------------
    # NEW COVARIANCE-BASED NOISE REALISATION FOR EACH GENERATION
    # --------------------------------------------------------

    noise_data = generate_covariance_based_noise(
        covariance=noise_covariance,
        n_generate=n_noise_samples,
        n_times=n_times,
        random_state=generation_seed,
        average_reference=True,
    )

    if (
        noise_data.ndim != 3
        or noise_data.shape[1:]
        != (n_channels, n_times)
    ):
        raise ValueError(
            f"Invalid noise shape: "
            f"{noise_data.shape}"
        )

    print("Noise:", noise_data.shape)

    # --------------------------------------------------------
    # ERP DATASET
    # --------------------------------------------------------

    metadata = generate_erp_dataset(
        lf_path=leadfield_file,
        out_dir=output_dir,
        source_indices=source_indices,
        noise_data=noise_data,
        n_trials_per_source=n_trials_per_source,
        n_times=n_times,
        sfreq=sfreq,
        erp_window=scaling_window,
        snr_db_range=snr_db_range,
        random_state=generation_seed,
        overwrite=overwrite_dataset,
    )

    if len(metadata) != expected_samples:
        raise RuntimeError(
            f"{gen_name}: expected "
            f"{expected_samples} samples, "
            f"got {len(metadata)}"
        )

    print(
        f"Sources={len(source_indices)} | "
        f"Trials/source={n_trials_per_source} | "
        f"Samples={len(metadata)}"
    )


# ============================================================
# SUMMARY
# ============================================================

samples_per_generation = (
    n_training_sources
    * n_trials_per_source
)

total_samples = (
    samples_per_generation
    * n_generations
)

print(f"\n{'=' * 60}")
print("ALL GENERATIONS COMPLETED")
print(f"{'=' * 60}")

print("Subject          :", gen_subject)
print("LF mode          :", lf_mode)
print("Training sources :", n_training_sources)
print("Trials/source    :", n_trials_per_source)
print("Generations      :", n_generations)
print("Samples/gen      :", samples_per_generation)
print("Total samples    :", total_samples)
print("Output           :", dataset_root)