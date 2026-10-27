import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from os import path, makedirs
from nilearn import plotting
from nilearn.datasets import (
    fetch_atlas_yeo_2011,
    fetch_atlas_harvard_oxford,
)
from scipy.spatial import cKDTree


YEO7_LABELS = {
    0: "Background",
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Salience / Ventral Attention",
    5: "Limbic",
    6: "Control",
    7: "Default",
}


YEO17_LABELS = {
    0: "Background",
    1: "Visual Central (Visual A)",
    2: "Visual Peripheral (Visual B)",
    3: "Somatomotor A",
    4: "Somatomotor B",
    5: "Dorsal Attention A",
    6: "Dorsal Attention B",
    7: "Salience / Ventral Attention A",
    8: "Salience / Ventral Attention B",
    9: "Limbic A",
    10: "Limbic B",
    11: "Control C",
    12: "Control A",
    13: "Control B",
    14: "Temporal Parietal",
    15: "Default C",
    16: "Default A",
    17: "Default B",
}

# Yeo labels:
# https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/
# brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/


def load_nilearn_atlas(atlas_name="yeo7"):
    atlas_name = atlas_name.lower()

    if atlas_name == "yeo7":
        atlas = fetch_atlas_yeo_2011(
            n_networks=7,
            thickness="thin",
            verbose=False,
        )
        img = ensure_3d_img(atlas.maps)
        labels = YEO7_LABELS

    elif atlas_name == "yeo17":
        atlas = fetch_atlas_yeo_2011(
            n_networks=17,
            thickness="thin",
            verbose=False,
        )
        img = ensure_3d_img(atlas.maps)
        labels = YEO17_LABELS

    elif atlas_name == "ho48":
        atlas = fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm",
            verbose=False,
        )
        img = ensure_3d_img(atlas.maps)
        labels = {
            i: lab
            for i, lab in enumerate(atlas.labels)
        }

    elif atlas_name == "ho69":
        cort = fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm",
            verbose=False,
        )
        sub = fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-2mm",
            verbose=False,
        )

        cort_img = ensure_3d_img(cort.maps)
        sub_img = ensure_3d_img(sub.maps)

        cort_data = cort_img.get_fdata().astype(int)
        sub_data = sub_img.get_fdata().astype(int)

        offset = len(cort.labels) - 1  # exclude background
        merged = cort_data.copy()

        mask = (merged == 0) & (sub_data > 0)
        merged[mask] = sub_data[mask] + offset

        img = nib.Nifti1Image(
            merged.astype(np.int16),
            cort_img.affine,
        )

        labels = {
            i: lab
            for i, lab in enumerate(cort.labels)
        }

        for i, lab in enumerate(sub.labels):
            if i == 0:
                continue

            labels[i + offset] = lab

    else:
        raise ValueError(
            "atlas_name must be one of: "
            "'yeo7', 'yeo17', 'ho48', 'ho69'"
        )

    data = img.get_fdata().astype(int)

    # For nearest valid atlas region
    label_mask = data > 0

    label_voxels_ijk = np.argwhere(label_mask)

    label_voxels_mm = nib.affines.apply_affine(
        img.affine,
        label_voxels_ijk,
    )

    label_values = data[label_mask]

    tree = cKDTree(label_voxels_mm)

    return {
        "name": atlas_name,
        "img": img,
        "data": data,
        "affine": img.affine,
        "inv_affine": np.linalg.inv(img.affine),
        "labels": labels,
        "tree": tree,
        "label_voxels_ijk": label_voxels_ijk,
        "label_voxels_mm": label_voxels_mm,
        "label_values": label_values,
    }


def query_atlas_coord(coords, atlas, threshold_mm=None):
    coords = np.asarray(coords, dtype=float)
    single_input = coords.ndim == 1
    coords = coords.reshape(-1, 3)

    labels = atlas["labels"]
    tree = atlas["tree"]
    label_voxels_ijk = atlas["label_voxels_ijk"]
    label_values = atlas["label_values"]
    data_shape = atlas["data"].shape[:3]

    results = []

    for coord in coords:
        distance, nn_idx = tree.query(coord, k=1)
        distance = float(distance)

        if threshold_mm is not None and distance > threshold_mm:
            region = "Unknown"
            region_id = best_voxel_id = -1
            best_voxel_ijk = [-1, -1, -1]

        else:
            best_voxel_ijk = label_voxels_ijk[nn_idx].astype(int)
            region_id = int(label_values[nn_idx])
            region = labels.get(region_id, f"Label_{region_id}")

            best_voxel_id = int(
                np.ravel_multi_index(
                    tuple(best_voxel_ijk),
                    data_shape,
                )
            )

            best_voxel_ijk = best_voxel_ijk.tolist()

        results.append({
            "atlas": atlas["name"],
            "region": region,
            "region_id": region_id,
            "distance_mm": round(distance, 3),
            "best_voxel_id": best_voxel_id,
            "best_voxel_ijk": best_voxel_ijk,
        })

    df = pd.DataFrame(results)
    return df.iloc[0].to_dict() if single_input else df


def ensure_3d_img(x):
    img = (
        x
        if isinstance(x, nib.Nifti1Image)
        else nib.load(x)
    )

    data = img.get_fdata()

    if data.ndim > 3:
        data = np.squeeze(data)

    if data.ndim > 3:
        data = data[..., 0]

    return nib.Nifti1Image(
        data.astype(np.int16),
        img.affine,
    )

