import os
import sys
from os import path

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from tqdm import tqdm
import argparse

project_path = path.dirname(path.dirname(path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.data_utils import get_run_from_filename, load_data_from_subject
from utils.nilearn_atlas import load_nilearn_atlas, query_atlas_coord
from utils.esolo_helpers import (
    preprocess_real, 
    load_esolo_model, 
    predict_source_distribution)


# ============================================================
# ARGUMENTS
# ============================================================
SUBJECTS_CHOICE = [ "sub-16", "sub-21", "sub-23", "sub-27",  "sub-33"]

parser = argparse.ArgumentParser()

parser.add_argument("--subject", default="all", choices=["all"] + SUBJECTS_CHOICE)
parser.add_argument("--loss_mode", default="leadfield", choices=["gaussian", "leadfield"])
parser.add_argument("--data_mode", default="syn_cov", choices=["syn_cov", "syn_random"])
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

selected_subject = args.subject
loss_mode, data_mode = args.loss_mode, args.data_mode
debug = args.debug

subject_list = (
    SUBJECTS_CHOICE
    if selected_subject == "all"
    else [selected_subject]
)

# ============================================================
# CONFIGURATION
# ============================================================
project_root = path.dirname(path.dirname(path.abspath(__file__)))

lf_mode = "vol"
model_name = "ESoLo"
leadfield_weight = 0.5
train_mode = f"{loss_mode}_{leadfield_weight}" if loss_mode == "leadfield" else loss_mode

atlas_names = ["yeo7", "yeo17", "ho48", "ho69"]
atlas_threshold_mm = 5.0

real_data_path = path.join(
    project_root, "data", "ccep_coreg", "derivatives", "epochs"
)
stim_info_path = path.join(
    project_root, "info", "ccep_coreg_stimulation_information.xlsx"
)
model_root = path.join(
    project_root, "results", "dl_train_gaussian_cov", lf_mode
)

output_root = path.join(
    project_root, "results", "dl_real_evaluation",
    lf_mode, model_name, data_mode,  loss_mode
)
if debug:
    output_root = path.join(output_root, "debug")

os.makedirs(output_root, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD STIMULATION METADATA
# ============================================================
df_stim = pd.read_excel(stim_info_path)
df_stim["subject"] = df_stim["subject"].astype(str)
df_stim["run"] = (
    df_stim["file"]
    .astype(str)
    .apply(get_run_from_filename)
)

stim_cols = [
    "subject", "file", "run",
    "stim_channel", "stim_intensity",
    "stim_x", "stim_y", "stim_z",
]

for atlas_name in atlas_names:
    stim_cols += [
        f"{atlas_name}_region",
        f"{atlas_name}_distance_mm",
    ]

stim_cols = [c for c in stim_cols if c in df_stim.columns]

df_stim = (
    df_stim[stim_cols]
    .drop_duplicates(["subject", "run"])
    .reset_index(drop=True)
)


# ============================================================
# LOAD ATLASES
# Same atlas utilities as numerical pipeline
# ============================================================
atlases = {
    name: load_nilearn_atlas(name)
    for name in atlas_names
}


# ============================================================
# EVALUATE SUBJECT-SPECIFIC MODELS
# ============================================================
all_predictions = []
subject_summaries = []

print("\n" + "=" * 80)
print("ESoLo REAL-DATA + ATLAS EVALUATION")
print("=" * 80)
print("Device     :", device)
print("Subject    :", selected_subject)
print("Data mode  :", data_mode)
print("Loss mode  :", loss_mode)
print("Train mode :", train_mode)
print("Output     :", output_root)


for subject in subject_list:
    print(f"\n{'=' * 70}")
    print(subject)
    print("=" * 70)

    checkpoint_path = path.join(
        model_root,
        subject,
        model_name,
        data_mode,
        train_mode,
        *( ["debug"] if debug else [] ),
        "best_model.pt",
    )

    if not path.isfile(checkpoint_path):
        print(f"Skip: checkpoint not found: {checkpoint_path}")
        continue

    real_eeg_dir = path.join(
        real_data_path,
        subject,
        "eeg",
    )

    if not path.isdir(real_eeg_dir):
        print(f"Skip: real EEG folder not found: {real_eeg_dir}")
        continue

    real_files = natsorted(
        f for f in os.listdir(real_eeg_dir)
        if f.endswith(".npy")
    )

    if not real_files:
        print(f"Skip: no real .npy files for {subject}")
        continue

    model, checkpoint, source_pos, n_channels, n_times = (
        load_esolo_model(checkpoint_path, device)
    )

    subject_rows = []

    for file in tqdm(real_files, desc=f"Predict {subject}"):
        run = get_run_from_filename(file)

        stim = df_stim[
            (df_stim["subject"] == subject)
            & (df_stim["run"] == run)
        ]

        if stim.empty:
            print(f"Skip {subject}/{run}: target not found")
            continue

        stim_row = stim.iloc[0]

        true_xyz = stim_row[
            ["stim_x", "stim_y", "stim_z"]
        ].to_numpy(dtype=np.float32)

        try:
            data = np.asarray(
                load_data_from_subject(
                    real_data_path,
                    subject,
                    run,
                    mode="real",
                ),
                dtype=np.float32,
            )
        except Exception as error:
            print(f"Skip {subject}/{run}: {error}")
            continue

        if (
            data.ndim != 3
            or data.shape[1] != n_channels
            or data.shape[2] != n_times
        ):
            print(
                f"Skip {subject}/{run}: shape={data.shape}, "
                f"expected (*,{n_channels},{n_times})"
            )
            continue

        eeg = torch.from_numpy(
            preprocess_real(data)
        ).to(device)

        pred = predict_source_distribution(
            model,
            eeg,
            source_pos,
        )

        pred_idx = int(pred["pred_idx"][0].item())

        pred_xyz = (
            pred["pred_xyz"][0]
            .detach().cpu().numpy()
        )

        expected_xyz = (
            pred["expected_xyz"][0]
            .detach().cpu().numpy()
        )

        confidence = float(
            pred["confidence"][0].item()
        )

        spatial_dispersion_mm = float(
            pred["spatial_dispersion_mm"][0].item()
        )

        dist_mm = float(
            np.linalg.norm(pred_xyz - true_xyz)
        )

        dist_expected_mm = float(
            np.linalg.norm(expected_xyz - true_xyz)
        )

        row = {
            "subject": subject,
            "file": file,
            "run": run,
            "model_name": model_name,
            "lf_mode": lf_mode,
            "train_mode": train_mode,
            "data_mode": data_mode,
            "fold": int(checkpoint.get("fold", -1)),
            "epoch": int(checkpoint.get("epoch", -1)),

            "true_x": float(true_xyz[0]),
            "true_y": float(true_xyz[1]),
            "true_z": float(true_xyz[2]),

            "pred_source_idx": pred_idx,
            "pred_x": float(pred_xyz[0]),
            "pred_y": float(pred_xyz[1]),
            "pred_z": float(pred_xyz[2]),

            "expected_x": float(expected_xyz[0]),
            "expected_y": float(expected_xyz[1]),
            "expected_z": float(expected_xyz[2]),

            "dist_mm": dist_mm,
            "dist_expected_mm": dist_expected_mm,
            "confidence": confidence,
            "spatial_dispersion_mm": spatial_dispersion_mm,
        }

        for col in stim.columns:
            if col not in {"subject", "file", "run"}:
                row[col] = stim_row[col]

        subject_rows.append(row)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not subject_rows:
        print(f"No valid predictions for {subject}")
        continue

    df_subject = pd.DataFrame(subject_rows)

    # ========================================================
    # ATLAS QUERY
    # Same matching logic as numerical pipeline
    # ========================================================
    pred_xyz_all = df_subject[
        ["pred_x", "pred_y", "pred_z"]
    ].to_numpy(float)

    for atlas_name in atlas_names:
        atlas = atlases[atlas_name]

        atlas_results = [
            query_atlas_coord(
                xyz,
                atlas,
                threshold_mm=atlas_threshold_mm,
            )
            for xyz in tqdm(
                pred_xyz_all,
                desc=f"{subject}-{atlas_name}",
                leave=False,
            )
        ]

        pred_col = f"pred_{atlas_name}_region"
        dist_col = f"pred_{atlas_name}_distance_mm"
        stim_col = f"{atlas_name}_region"
        match_col = f"{atlas_name}_match"

        df_subject[pred_col] = [
            r["region"] for r in atlas_results
        ]
        df_subject[dist_col] = [
            r["distance_mm"] for r in atlas_results
        ]

        if stim_col not in df_subject.columns:
            df_subject[match_col] = np.nan
            continue

        pred_region = (
            df_subject[pred_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        stim_region = (
            df_subject[stim_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df_subject[match_col] = (
            pred_region.eq(stim_region)
            & pred_region.ne("")
        )

    # ========================================================
    # SUBJECT SUMMARY
    # ========================================================
    summary = {
        "subject": subject,
        "model_name": model_name,
        "lf_mode": lf_mode,
        "train_mode": train_mode,
        "data_mode": data_mode,
        "n_predictions": len(df_subject),

        "distance_mean_mm": df_subject["dist_mm"].mean(),
        "distance_std_mm": df_subject["dist_mm"].std(ddof=1),
        "distance_median_mm": df_subject["dist_mm"].median(),
        "distance_min_mm": df_subject["dist_mm"].min(),
        "distance_max_mm": df_subject["dist_mm"].max(),
        "distance_q1_mm": df_subject["dist_mm"].quantile(0.25),
        "distance_q3_mm": df_subject["dist_mm"].quantile(0.75),

        "spatial_dispersion_mean_mm": (
            df_subject["spatial_dispersion_mm"].mean()
        ),
        "spatial_dispersion_std_mm": (
            df_subject["spatial_dispersion_mm"].std(ddof=1)
        ),
        "spatial_dispersion_median_mm": (
            df_subject["spatial_dispersion_mm"].median()
        ),
        "spatial_dispersion_min_mm": (
            df_subject["spatial_dispersion_mm"].min()
        ),
        "spatial_dispersion_max_mm": (
            df_subject["spatial_dispersion_mm"].max()
        ),

        "confidence_mean": df_subject["confidence"].mean(),
    }

    for atlas_name in atlas_names:
        match_col = f"{atlas_name}_match"

        if match_col in df_subject.columns:
            valid = df_subject[match_col].dropna()

            summary[f"{atlas_name}_accuracy_percent"] = (
                100.0 * valid.mean()
                if len(valid) > 0
                else np.nan
            )

    subject_summaries.append(summary)
    all_predictions.append(df_subject)

    # Per-subject output.
    subject_output_file = path.join(
        output_root,
        f"{subject}_real_data_predictions.xlsx",
    )

    with pd.ExcelWriter(
        subject_output_file,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame([summary]).round(4).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        df_subject.to_excel(
            writer,
            sheet_name="Predictions",
            index=False,
        )

    print(
        f"{subject}"
        f" | Error {summary['distance_mean_mm']:.2f} mm"
        f" | Dispersion {summary['spatial_dispersion_mean_mm']:.2f} mm"
        f" | HO48 {summary.get('ho48_accuracy_percent', np.nan):.2f}%"
    )


# ============================================================
# COMBINE ALL SUBJECTS
# ============================================================
if not all_predictions:
    raise RuntimeError(
        "No valid ESoLo real-data predictions generated."
    )

df_all = pd.concat(
    all_predictions,
    ignore_index=True,
)

df_subject_summary = pd.DataFrame(
    subject_summaries
)


# ============================================================
# OVERALL SUMMARY
# Calculated from all predictions, not mean of subject means.
# ============================================================
overall = {
    "model_name": model_name,
    "lf_mode": lf_mode,
    "train_mode": train_mode,
    "data_mode": data_mode,
    "n_subjects": df_all["subject"].nunique(),
    "n_predictions": len(df_all),

    "distance_mean_mm": df_all["dist_mm"].mean(),
    "distance_std_mm": df_all["dist_mm"].std(ddof=1),
    "distance_median_mm": df_all["dist_mm"].median(),
    "distance_min_mm": df_all["dist_mm"].min(),
    "distance_max_mm": df_all["dist_mm"].max(),
    "distance_q1_mm": df_all["dist_mm"].quantile(0.25),
    "distance_q3_mm": df_all["dist_mm"].quantile(0.75),

    "spatial_dispersion_mean_mm": (df_all["spatial_dispersion_mm"].mean()  ),
    "spatial_dispersion_std_mm": (df_all["spatial_dispersion_mm"].std(ddof=1) ),
    "spatial_dispersion_median_mm": ( df_all["spatial_dispersion_mm"].median()),
    "spatial_dispersion_min_mm": ( df_all["spatial_dispersion_mm"].min() ),
    "spatial_dispersion_max_mm": ( df_all["spatial_dispersion_mm"].max() ),
    "confidence_mean": df_all["confidence"].mean(),
}

for atlas_name in atlas_names:
    match_col = f"{atlas_name}_match"

    if match_col in df_all.columns:
        valid = df_all[match_col].dropna()

        overall[f"{atlas_name}_accuracy_percent"] = (
            100.0 * valid.mean()
            if len(valid) > 0
            else np.nan
        )

df_overall = pd.DataFrame([overall])


# ============================================================
# SAVE COMBINED OUTPUT
# ============================================================
combined_filename = (
    "all_subjects_real_data_summary.xlsx"
    if selected_subject == "all"
    else f"{selected_subject}_real_data_summary.xlsx"
)
combined_output_file = path.join(output_root, combined_filename)

with pd.ExcelWriter(combined_output_file, engine="openpyxl") as writer:
    df_overall.round(4).to_excel(writer, sheet_name="Overall_Summary", index=False)
    df_subject_summary.round(4).to_excel(
        writer, sheet_name="Subject_Summary", index=False
    )
    df_all.to_excel(writer, sheet_name="All_Predictions", index=False)

    for subject, df_subject in df_all.groupby("subject", sort=False):
        df_subject.to_excel(writer, sheet_name=subject[:31], index=False)


# ============================================================
# FINAL PRINT
# ============================================================
print("\n" + "=" * 80)
print("OVERALL REAL-DATA RESULT")
print("=" * 80)
print("Subjects    :", overall["n_subjects"])
print("Predictions :", overall["n_predictions"])
print(
    f"Error       : {overall['distance_mean_mm']:.2f} ± "
    f"{overall['distance_std_mm']:.2f} mm"
)
print(
    f"Dispersion  : {overall['spatial_dispersion_mean_mm']:.2f} ± "
    f"{overall['spatial_dispersion_std_mm']:.2f} mm"
)

for atlas_name in atlas_names:
    key = f"{atlas_name}_accuracy_percent"
    if key in overall:
        print(f"{atlas_name.upper():<12}: {overall[key]:.2f}%")

print("\nSaved:", combined_output_file)