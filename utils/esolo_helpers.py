# ============================================================
# IMPORT
# ============================================================
import random
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

from os import path
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from nilearn import plotting

from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

project_path = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(project_path)

from utils.models.ESoLo import ESoLo


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD LEADFIELD
# ============================================================
def load_leadfield(lf_file):
    lf = loadmat(lf_file)["lf"]

    return {
        "leadfield": lf["leadfield"][0, 0].astype(np.float32),
        "pos": lf["pos"][0, 0].astype(np.float32),
        "ori": lf["orientation"][0, 0].astype(np.float32),
        "chanlocs": lf["chanlocs"][0, 0],
    }


# ============================================================
# K-FOLD SPLITS
# ============================================================
def create_kfold_splits(
    run_folders,
    n_splits=5,
    seed=42,
):
    run_folders = list(run_folders)

    if len(run_folders) < n_splits:
        raise ValueError(
            f"Runs ({len(run_folders)}) < folds ({n_splits})"
        )

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    return [
        (
            [run_folders[i] for i in train_idx],
            [run_folders[i] for i in val_idx],
        )
        for train_idx, val_idx
        in kfold.split(run_folders)
    ]


# ============================================================
# BUILD SYNTHETIC DATA INDEX
# ============================================================

def load_synthetic_index(data_root, run_folders):
    """
    Build sample index without loading EEG into RAM.

    New storage:
        one .npy per source
        shape = (n_trials, n_channels, n_times)

    Returns
    -------
    samples : list[dict]
        One item per individual trial.
    """

    samples = []

    for run in run_folders:
        run_dir = path.join(data_root, run)
        metadata_file = path.join(run_dir, "metadata.xlsx")

        if not path.isfile(metadata_file):
            raise FileNotFoundError(
                f"Metadata tidak ditemukan: {metadata_file}"
            )

        metadata = pd.read_excel(metadata_file)

        required = {
            "eeg_file",
            "trial_idx",
            "source_idx",
            "source_x",
            "source_y",
            "source_z",
        }

        missing = required - set(metadata.columns)

        if missing:
            raise ValueError(
                f"{run}: missing metadata columns "
                f"{sorted(missing)}"
            )

        for row in metadata.to_dict("records"):
            eeg_file = path.join(
                run_dir,
                row["eeg_file"],
            )

            if not path.isfile(eeg_file):
                raise FileNotFoundError(
                    f"EEG file tidak ditemukan: {eeg_file}"
                )

            samples.append({
                "eeg_file": eeg_file,
                "trial_idx": int(row["trial_idx"]),
                "source_idx": int(row["source_idx"]),
                "source_x": float(row["source_x"]),
                "source_y": float(row["source_y"]),
                "source_z": float(row["source_z"]),
                "generation": run,
            })

    if not samples:
        raise RuntimeError(
            f"No synthetic samples found in: {data_root}"
        )

    print(
        f"Indexed {len(samples):,} trials from "
        f"{len(run_folders)} generations"
    )

    return samples


# ============================================================
# LAZY EEG DATASET
# ============================================================

class EEGLazyGaussianDataset(Dataset):

    def __init__(
        self,
        samples,
        source_pos,
        sigma_mm=5.0,
        radius_mm=10.0,
    ):
        self.samples = samples

        self.source_pos = torch.as_tensor(
            source_pos,
            dtype=torch.float32,
        )

        self.sigma_mm = float(sigma_mm)
        self.radius_mm = radius_mm

        # ----------------------------------------------------
        # CHECK FIRST FILE + GET SHAPE
        # ----------------------------------------------------

        first = self.samples[0]

        first_data = np.load(
            first["eeg_file"],
            mmap_mode="r",
        )

        if first_data.ndim != 3:
            raise ValueError(
                f"{first['eeg_file']}: expected "
                f"(trials, channels, time), "
                f"got {first_data.shape}"
            )

        self.n_channels = first_data.shape[1]
        self.n_times = first_data.shape[2]

        del first_data

        # ----------------------------------------------------
        # PRECOMPUTE TARGET ONLY FOR UNIQUE TRAINING SOURCES
        # ----------------------------------------------------

        unique_source_idx = sorted({
            sample["source_idx"]
            for sample in self.samples
        })

        unique_source_idx = torch.as_tensor(
            unique_source_idx,
            dtype=torch.long,
        )

        target_xyz = self.source_pos[
            unique_source_idx
        ]

        dist = torch.cdist(
            target_xyz,
            self.source_pos,
        )

        logits = -(
            dist.square()
            / (2.0 * self.sigma_mm**2)
        )

        if self.radius_mm is not None:
            mask = dist <= self.radius_mm

            nearest = dist.argmin(dim=1)

            mask[
                torch.arange(len(dist)),
                nearest,
            ] = True

            logits = logits.masked_fill(
                ~mask,
                -1e9,
            )

        target_prob = torch.softmax(
            logits,
            dim=1,
        )

        self.target_prob = {
            int(source_idx):
            target_prob[i]
            for i, source_idx
            in enumerate(unique_source_idx.tolist())
        }

        # mmap cache:
        # one dictionary per Dataset/worker
        self._mmap_cache = {}

        print(
            "Unique targets :",
            len(self.target_prob),
        )
        print(
            "EEG shape      :",
            (self.n_channels, self.n_times),
        )

    def __len__(self):
        return len(self.samples)

    def _load_file(self, file_path):
        """
        Keep memory-mapped file handle cached.

        mmap does not load the entire file into RAM.
        """

        if file_path not in self._mmap_cache:
            self._mmap_cache[file_path] = np.load(
                file_path,
                mmap_mode="r",
            )

        return self._mmap_cache[file_path]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        eeg_trials = self._load_file(
            sample["eeg_file"]
        )

        trial_idx = sample["trial_idx"]

        if not 0 <= trial_idx < eeg_trials.shape[0]:
            raise IndexError(
                f"trial_idx={trial_idx} outside "
                f"{eeg_trials.shape}"
            )

        # Copy only ONE trial from mmap
        eeg = np.asarray(
            eeg_trials[trial_idx],
            dtype=np.float32,
        ).copy()

        # ----------------------------------------------------
        # SAME PREPROCESSING AS BEFORE
        # ----------------------------------------------------

        # Average reference
        eeg -= eeg.mean(
            axis=0,
            keepdims=True,
        )

        # Global z-score
        eeg -= eeg.mean()
        eeg /= eeg.std() + 1e-8

        source_idx = sample["source_idx"]

        target_xyz = torch.tensor(
            [
                sample["source_x"],
                sample["source_y"],
                sample["source_z"],
            ],
            dtype=torch.float32,
        )

        return {
            "eeg": torch.from_numpy(eeg),
            "target_prob": self.target_prob[source_idx],
            "target_xyz": target_xyz,
            "source_idx": torch.tensor(
                source_idx,
                dtype=torch.long,
            ),
        }


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_lazy_dataloaders(
    train_samples,
    val_samples,
    source_pos,
    batch_size=64,
    sigma_mm=5.0,
    radius_mm=10.0,
    num_workers=4,
):
    train_dataset = EEGLazyGaussianDataset(
        samples=train_samples,
        source_pos=source_pos,
        sigma_mm=sigma_mm,
        radius_mm=radius_mm,
    )

    val_dataset = EEGLazyGaussianDataset(
        samples=val_samples,
        source_pos=source_pos,
        sigma_mm=sigma_mm,
        radius_mm=radius_mm,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )



# ============================================================
# GAUSSIAN SOFT TARGET LOSS
# ============================================================

class GaussianSoftTargetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_logits,
        target_prob,
    ):
        """
        Parameters
        ----------
        pred_logits : torch.Tensor
            Shape (batch, n_sources).

        target_prob : torch.Tensor
            Gaussian soft target distribution,
            shape (batch, n_sources).

        Returns
        -------
        dict
            loss and classification_loss.
        """

        pred_log_prob = F.log_softmax(pred_logits, dim=1)

        classification_loss = -(
            target_prob * pred_log_prob
        ).sum(dim=1).mean()

        # Entropy target Gaussian
        target_entropy = -(
            target_prob * torch.log(target_prob.clamp_min(1e-12))
        ).sum(dim=1).mean()

        # KL(q||p) = CE - H(q)
        kl_divergence = classification_loss - target_entropy

        return {
            "loss": classification_loss,
            "classification_loss": classification_loss.detach(),
            "target_entropy": target_entropy.detach(),
            "kl_di  vergence": kl_divergence.detach(),
        }
    
    
# ============================================================
# GAUSSIAN + FULL-ORIENTATION LEADFIELD LOSS
# ============================================================
class GaussianLeadfieldLoss(nn.Module):
    def __init__(
        self,
        leadfield,
        classification_weight=0.5,
        leadfield_weight=0.5,
    ):
        super().__init__()

        leadfield = torch.as_tensor(
            leadfield,
            dtype=torch.float32,
        )

        if leadfield.ndim != 3:
            raise ValueError(
                "Expected leadfield shape "
                "(channels, sources, orientations)"
            )

        # (channels, sources, orientations)
        # -> (sources, channels, orientations)
        leadfield = leadfield.permute(
            1, 0, 2
        ).contiguous()

        self.register_buffer(
            "leadfield",
            leadfield,
        )

        self.classification_weight = float(
            classification_weight
        )
        self.leadfield_weight = float(
            leadfield_weight
        )

    def _source_to_topography(self, probability):
        # probability: (batch, sources)
        # leadfield : (sources, channels, orientations)
        # result    : (batch, channels, orientations)
        return torch.einsum(
            "bs,sco->bco",
            probability,
            self.leadfield,
        )

    @staticmethod
    def _normalize_topography(pattern):
        # Average reference per orientation.
        pattern = pattern - pattern.mean(
            dim=1,
            keepdim=True,
        )

        # Normalize each orientation independently.
        return F.normalize(
            pattern,
            p=2,
            dim=1,
            eps=1e-8,
        )

    def forward(self, pred_logits, target_prob):
        pred_log_prob = F.log_softmax(
            pred_logits,
            dim=1,
        )
        pred_prob = torch.softmax(
            pred_logits,
            dim=1,
        )

        classification_loss = -(
            target_prob * pred_log_prob
        ).sum(dim=1).mean()

        target_entropy = -(
            target_prob * torch.log(target_prob.clamp_min(1e-12))
        ).sum(dim=1).mean()

        kl_divergence = classification_loss - target_entropy

        pred_pattern = self._source_to_topography(
            pred_prob
        )
        target_pattern = self._source_to_topography(
            target_prob
        )

        pred_pattern = self._normalize_topography(
            pred_pattern
        )
        target_pattern = self._normalize_topography(
            target_pattern
        )

        # Compare all orientation combinations.
        # pred   : (batch, channels, pred_orientation)
        # target : (batch, channels, target_orientation)
        # result : (batch, pred_orientation, target_orientation)
        cosine_matrix = torch.einsum(
            "bco,bcp->bop",
            pred_pattern,
            target_pattern,
        )

        # Polarity can be reversed, so use absolute similarity.
        cosine_matrix = cosine_matrix.abs()

        # Strongest orientation combination per sample.
        strongest_similarity = cosine_matrix.amax(
            dim=(1, 2)
        )

        leadfield_loss = (
            1.0 - strongest_similarity
        ).mean()

        total_loss = (
            self.classification_weight
            * classification_loss
            + self.leadfield_weight
            * leadfield_loss
        )

        return {
            "loss": total_loss,
            "classification_loss": classification_loss.detach(),
            "target_entropy": target_entropy.detach(),
            "kl_divergence": kl_divergence.detach(),
            "leadfield_loss": leadfield_loss.detach(),
            "leadfield_similarity": strongest_similarity.mean().detach(),
        }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    metric_sums = {}
    n_samples = 0

    for batch in loader:
        eeg = batch["eeg"].to(
            device,
            non_blocking=True,
        )

        target_prob = batch["target_prob"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        pred_logits = model(eeg)

        losses = criterion(
            pred_logits,
            target_prob,
        )

        losses["loss"].backward()
        optimizer.step()

        batch_size = eeg.size(0)
        n_samples += batch_size

        for metric_name, metric_value in losses.items():
            if torch.is_tensor(metric_value):
                metric_value = metric_value.item()

            metric_sums[metric_name] = (
                metric_sums.get(metric_name, 0.0)
                + float(metric_value) * batch_size
            )

    if n_samples == 0:
        raise RuntimeError(
            "DataLoader contains no samples."
        )

    return {
        metric_name: metric_sum / n_samples
        for metric_name, metric_sum
        in metric_sums.items()
    }


# ============================================================
# VALIDATE ONE EPOCH
# ============================================================
@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    source_pos,
    device,
):
    model.eval()

    source_pos = torch.as_tensor(
        source_pos,
        dtype=torch.float32,
        device=device,
    )

    metric_sums = {}

    total_distance = 0.0
    total_correct = 0
    n_samples = 0

    for batch in loader:
        eeg = batch["eeg"].to(
            device,
            non_blocking=True,
        )

        target_prob = batch["target_prob"].to(
            device,
            non_blocking=True,
        )

        target_xyz = batch["target_xyz"].to(
            device,
            non_blocking=True,
        )

        pred_logits = model(eeg)

        losses = criterion(
            pred_logits,
            target_prob,
        )

        pred_source_idx = pred_logits.argmax(dim=1)
        pred_xyz = source_pos[pred_source_idx]

        distance_mm = torch.linalg.vector_norm(
            pred_xyz - target_xyz,
            dim=1,
        )

        batch_size = eeg.size(0)
        n_samples += batch_size

        for metric_name, metric_value in losses.items():
            if torch.is_tensor(metric_value):
                metric_value = metric_value.item()

            metric_sums[metric_name] = (
                metric_sums.get(metric_name, 0.0)
                + float(metric_value) * batch_size
            )

        total_distance += distance_mm.sum().item()

        # Correct when argmax falls inside Gaussian target support.
        batch_idx = torch.arange(batch_size, device=device)
        correct = target_prob[batch_idx, pred_source_idx] > 0
        total_correct += correct.sum().item()

    if n_samples == 0:
        raise RuntimeError(
            "DataLoader contains no samples."
        )

    metrics = {
        metric_name: metric_sum / n_samples
        for metric_name, metric_sum
        in metric_sums.items()
    }

    metrics.update({
        "distance_mm": (
            total_distance / n_samples
        ),
        "source_accuracy_percent": (
            100.0 * total_correct / n_samples
        ),
    })

    return metrics


# ============================================================
# RESULTS
# ============================================================
def build_epoch_result(
    fold,
    epoch,
    learning_rate,
    train_metrics,
    val_metrics,
):
    result = {
        "fold": fold,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": float(train_metrics["loss"]),
        "val_loss": float(val_metrics["loss"]),
        "val_distance_mm": float(val_metrics["distance_mm"]),
        "val_source_accuracy_percent": float(
            val_metrics["source_accuracy_percent"]
        ),
    }

    for name, value in train_metrics.items():
        if name != "loss" and pd.api.types.is_number(value):
            result[f"train_{name}"] = float(value)

    excluded = {"loss", "distance_mm", "source_accuracy_percent"}
    for name, value in val_metrics.items():
        if name not in excluded and pd.api.types.is_number(value):
            result[f"val_{name}"] = float(value)

    return result


def build_checkpoint(
    model,
    optimizer,
    fold,
    epoch,
    model_name,
    data_mode,
    syn_data_path,
    train_runs,
    val_runs,
    source_pos,
    n_channels,
    n_times,
    n_sources,
    hidden_size,
    attention_size,
    channel_reduction,
    dropout,
    gaussian_sigma_mm,
    gaussian_radius_mm,
    best_val_loss,
    best_val_distance_mm,
    best_val_source_accuracy_percent,
    loss_mode,
    classification_weight= None,
    leadfield_weight = None,
):
    return {
        "fold": fold,
        "epoch": epoch,
        "model_name": model_name,
        "data_mode": data_mode,
        "loss_mode": loss_mode,
        "classification_weight": classification_weight,
        "leadfield_weight": leadfield_weight,
        "synthetic_data_path": syn_data_path,

        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),

        "best_val_loss": best_val_loss,
        "best_val_distance_mm": best_val_distance_mm,
        "best_val_source_accuracy_percent":
            best_val_source_accuracy_percent,

        "n_channels": n_channels,
        "n_times": n_times,
        "n_sources": n_sources,
        "hidden_size": hidden_size,
        "attention_size": attention_size,
        "channel_reduction": channel_reduction,
        "dropout": dropout,

        "gaussian_sigma_mm": gaussian_sigma_mm,
        "gaussian_radius_mm": gaussian_radius_mm,
        "source_pos": source_pos,
        "train_runs": train_runs,
        "val_runs": val_runs,
    }


def build_history_summary(all_history_df):
    summary = (
        all_history_df.groupby("epoch", as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
            val_distance_mean=("val_distance_mm", "mean"),
            val_distance_std=("val_distance_mm", "std"),
            val_accuracy_mean=("val_source_accuracy_percent", "mean"),
            val_accuracy_std=("val_source_accuracy_percent", "std"),
            epoch_time_mean=("epoch_time_sec", "mean"),
            epoch_time_std=("epoch_time_sec", "std"),
            n_folds=("fold", "nunique"),
        )
    )

    std_columns = [
        "train_loss_std", "val_loss_std", "val_distance_std",
        "val_accuracy_std", "epoch_time_std",
    ]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def save_fold_outputs(
    result_path,
    fold_histories,
    all_history_df,
    history_summary,
    fold_results_df,
    training_info=None,
):
    output_file = path.join(result_path, "fold_histories.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, dataframe in fold_histories.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

        all_history_df.to_excel(
            writer, sheet_name="All_Folds_Long", index=False
        )
        history_summary.to_excel(
            writer, sheet_name="Mean_STD", index=False
        )
        fold_results_df.to_excel(
            writer, sheet_name="Fold_Summary", index=False
        )

        if training_info is not None:
            training_info.to_excel(
                writer, sheet_name="Training_Info", index=False
            )

    return output_file


# ============================================================
# PLOTING
# ============================================================
def plot_mean_std(
    x,
    series,
    ylabel,
    title,
    output_file,
):
    plt.figure(figsize=(10, 5))

    for mean, std, label in series:
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_mean_history(history_summary, result_path):
    epochs = history_summary["epoch"].to_numpy()

    plot_mean_std(
        x=epochs,
        series=[
            (
                history_summary["train_loss_mean"].to_numpy(),
                history_summary["train_loss_std"].to_numpy(),
                "Training loss",
            ),
            (
                history_summary["val_loss_mean"].to_numpy(),
                history_summary["val_loss_std"].to_numpy(),
                "Validation loss",
            ),
        ],
        ylabel="Loss",
        title="Training and validation loss across folds",
        output_file=path.join(result_path, "mean_folds_loss.png"),
    )

    plot_mean_std(
        x=epochs,
        series=[
            (
                history_summary["val_distance_mean"].to_numpy(),
                history_summary["val_distance_std"].to_numpy(),
                "Validation distance",
            )
        ],
        ylabel="Localisation error (mm)",
        title="Validation localisation error across folds",
        output_file=path.join(
            result_path, "mean_folds_val_distance.png"
        ),
    )


# ============================================================
# REAL PREPROCESSING
# ============================================================
def preprocess_real(data):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"Expected (trials, channels, time), got {data.shape}"
        )

    evoked = data.mean(axis=0)

    # Same preprocessing as synthetic training.
    evoked -= evoked.mean(axis=0, keepdims=True)
    evoked -= evoked.mean()
    evoked /= evoked.std() + 1e-8

    return evoked[None].astype(np.float32)


# ============================================================
# LOAD CHECKPOINT
# ============================================================
def load_esolo_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    n_channels = int(checkpoint["n_channels"])
    n_times = int(checkpoint.get("n_times", 1000))
    n_sources = int(checkpoint["n_sources"])

    source_pos = torch.as_tensor(
        checkpoint["source_pos"],
        dtype=torch.float32,
        device=device,
    )

    if source_pos.shape != (n_sources, 3):
        raise ValueError(
            f"source_pos={tuple(source_pos.shape)}, "
            f"expected ({n_sources}, 3)"
        )

    model = ESoLo(
        n_channels=n_channels,
        n_sources=n_sources,
        hidden_size=checkpoint.get("hidden_size", 64),
        attention_size=checkpoint.get("attention_size", 64),
        channel_reduction=checkpoint.get("channel_reduction", 4),
        dropout=checkpoint.get("dropout", 0.2),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, source_pos, n_channels, n_times


# ============================================================
# PREDICTION + SPATIAL DISPERSION
# ============================================================
@torch.no_grad()
def predict_esolo(model, eeg, source_pos):
    logits = model(eeg)

    if isinstance(logits, dict):
        if "source_logits" in logits:
            logits = logits["source_logits"]
        elif "output" in logits:
            logits = logits["output"]
        else:
            raise KeyError(
                f"Could not find logits in keys: {list(logits.keys())}"
            )

    prob = torch.softmax(logits, dim=1)
    pred_idx = prob.argmax(dim=1)
    pred_xyz = source_pos[pred_idx]
    expected_xyz = prob @ source_pos
    confidence = prob.max(dim=1).values

    # Probability-weighted spatial dispersion around expected centre.
    squared_distance = (
        source_pos.unsqueeze(0)
        - expected_xyz.unsqueeze(1)
    ).square().sum(dim=2)

    spatial_dispersion = torch.sqrt(
        (prob * squared_distance).sum(dim=1)
    )

    return {
        "pred_idx": pred_idx,
        "pred_xyz": pred_xyz,
        "expected_xyz": expected_xyz,
        "confidence": confidence,
        "spatial_dispersion_mm": spatial_dispersion,
    }



# ============================================================
# PREDICTION + SOURCE DISTRIBUTION
# ============================================================
@torch.no_grad()
def predict_source_distribution(model, eeg, source_pos):
    logits = model(eeg)

    if isinstance(logits, dict):
        if "source_logits" in logits:
            logits = logits["source_logits"]

        elif "output" in logits:
            logits = logits["output"]

        else:
            raise KeyError(f"Could not find logits in {list(logits.keys())}")

    # Full spatial probability distribution.
    prob = torch.softmax(logits, dim=1)

    # Point prediction.
    pred_idx = prob.argmax(dim=1)
    pred_xyz = source_pos[pred_idx]

    # Probability-weighted centre.
    expected_xyz = prob @ source_pos
    confidence = prob.max(dim=1).values

    # Probability-weighted RMS spatial dispersion.
    squared_distance = (source_pos.unsqueeze(0) - expected_xyz.unsqueeze(1)).square().sum(dim=2)
    spatial_dispersion = torch.sqrt((prob * squared_distance).sum(dim=1))

    return {
        "prob": prob,
        "pred_idx": pred_idx,
        "pred_xyz": pred_xyz,
        "expected_xyz": expected_xyz,
        "confidence": confidence,
        "spatial_dispersion_mm": spatial_dispersion,
    }


# ============================================================
# SMOOTH SOURCE DISTRIBUTION
# ============================================================

def plot_source_distribution(
    subject,
    run,
    source_pos,
    source_prob,
    true_xyz,
    pred_xyz,
    dist_mm,
    dispersion_mm,
    output_file,
    percentile=90,
    grid_size=200,
    sigma=7.0,
):
    source_pos = np.asarray(source_pos, dtype=float)
    source_prob = np.asarray(source_prob, dtype=float).squeeze()
    true_xyz = np.asarray(true_xyz, dtype=float)
    pred_xyz = np.asarray(pred_xyz, dtype=float)

    if len(source_prob) != len(source_pos):
        raise ValueError(
            f"Probability/source mismatch: " f"{source_prob.shape} vs {source_pos.shape}"
        )

    valid = np.isfinite(source_prob) & np.all(np.isfinite(source_pos), axis=1)
    xyz = source_pos[valid]
    values = source_prob[valid]
    threshold = np.percentile(values, percentile)
    mask = values >= threshold

    xyz = xyz[mask]
    values = values[mask]

    if len(values) < 3:
        print(f"Skip plot {subject}/{run}: " "not enough source points")
        return

    # Normalize for visualisation
    values = (values - values.min()) / (values.max() - values.min() + 1e-12)

    # --------------------------------------------------------
    # GLASS BRAIN
    # --------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))

    display = plotting.plot_glass_brain(
        None, display_mode="ortho", figure=fig, black_bg=False, colorbar=False
    )

    mappable = None
    brain_axes = [axis.ax for axis in display.axes.values()]

    for ax in brain_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 * 0.86, pos.y0, pos.width * 0.86, pos.height])

    # SMOOTH EACH 2D PROJECTION
    for axis in display.axes.values():

        ax = axis.ax
        direction = axis.direction

        if direction == "x":
            horizontal = xyz[:, 1]
            vertical = xyz[:, 2]
        elif direction == "y":
            horizontal = xyz[:, 0]
            vertical = xyz[:, 2]
        elif direction == "z":
            horizontal = xyz[:, 0]
            vertical = xyz[:, 1]
        else:
            continue

        # Continuous grid
        x_grid = np.linspace(horizontal.min(), horizontal.max(), grid_size)
        y_grid = np.linspace(vertical.min(), vertical.max(), grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        zz = griddata(
            points=np.column_stack([horizontal, vertical]),
            values=values,
            xi=(xx, yy),
            method="linear",
            fill_value=0.0,
        )

        # Gaussian smoothing
        zz = gaussian_filter(zz, sigma=sigma)

        # Remove extremely weak background
        if np.nanmax(zz) > 0:
            zz = zz / np.nanmax(zz)

        zz[zz < 0.05] = np.nan

        # Smooth probability cloud
        mappable = ax.contourf(
            xx, yy, zz, levels=np.linspace(0.05, 1.0, 20), cmap="hot", alpha=0.50, zorder=1
        )

    # STIM -> PREDICTION LINE
    for axis in display.axes.values():
        ax = axis.ax
        direction = axis.direction

        if direction == "x":
            x1, y1 = true_xyz[1], true_xyz[2]
            x2, y2 = pred_xyz[1], pred_xyz[2]
        elif direction == "y":
            x1, y1 = true_xyz[0], true_xyz[2]
            x2, y2 = pred_xyz[0], pred_xyz[2]
        elif direction == "z":
            x1, y1 = true_xyz[0], true_xyz[1]
            x2, y2 = pred_xyz[0], pred_xyz[1]
        else:
            continue

        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1.3, alpha=0.8, zorder=3)



    display.add_markers(true_xyz.reshape(1, 3), marker_color="black", marker_size=60)
    display.add_markers(pred_xyz.reshape(1, 3), marker_color="red", marker_size=55)

    #colorbar
    if mappable is not None:

        cbar_ax = fig.add_axes(
            [0.995, 0.1, 0.015, 0.75]  # left → makin besar makin kanan  # bottom  # width  # height
        )

        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Relative source probability", fontsize=11)
        cbar.ax.tick_params(labelsize=8)

    # legend and text
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="none",
            alpha=0.6,
            markersize=10,
            label="Source distribution",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="Stimulation",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=10,
            label="Argmax prediction",
        ),
        plt.Line2D([0], [0], color="gray", linewidth=1.3, label="Localisation error"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=12,
    )

    fig.text( 0.5,  0.015,
        (
            f"{subject} — {run}  |  "
            f"LE: {dist_mm:.2f} mm  |  "
            f"Dispersion: {dispersion_mm:.2f} mm"
        ),
        ha="center",
        va="center",
        fontsize=14,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
