# ============================================================
# IMPORT
# ============================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
import argparse

from torchinfo import summary
from os import path

project_path = path.dirname(path.dirname(path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)

from utils.esolo_helpers import *
from utils.models.ESoLo import ESoLo


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()

SUBJECTS_CHOICE = [ "sub-16", "sub-21", "sub-23", "sub-27",  "sub-33"]

parser.add_argument("--subject", default="sub-16" , choices = SUBJECTS_CHOICE)
parser.add_argument("--loss_mode", default="leadfield", choices=["gaussian", "leadfield"])
parser.add_argument("--data_mode", default="syn_cov", choices=["syn_cov", "syn_random"])
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

subject = args.subject
loss_mode, data_mode  = args.loss_mode, args.data_mode
debug =  args.debug

# ============================================================
# DEFAULT PARAMETERS
# ============================================================
model_name = "ESoLo"
lf_mode ="vol"
gaussian_sigma_mm, gaussian_radius_mm = 10.0, 15.0 #5.0, 10.0
seed, batch_size, patience = 42, 64, 10
n_folds, n_epochs = (3, 5) if debug else (5, 100)
learning_rate, weight_decay = 1e-3, 1e-4

hidden_size, attention_size = 64, 64
channel_reduction, dropout = 4, 0.2

classification_weight = None
leadfield_weight = None

if loss_mode == "leadfield":
    classification_weight = 0.5
    leadfield_weight = 0.5

    if abs(classification_weight + leadfield_weight - 1.0) > 1e-8:
        raise ValueError(
            f"classification_weight + leadfield_weight must equal 1.0 "
            f"(got {classification_weight + leadfield_weight:.4f})"
        )

    train_mode = f"{loss_mode}_{leadfield_weight}"
else:
    train_mode = loss_mode


# ============================================================
# PATHS
# ============================================================
project_root = path.dirname(path.dirname(path.abspath(__file__)))

syn_data_path = path.join(
    project_root, "data", data_mode, lf_mode, subject
)

lf_file = path.join(
    project_root, "data", "ccep_coreg_lf",
    f"{subject}_lf_{lf_mode}.mat",
)

if debug:
    result_path = path.join(
        project_root, "results", "dl_train_gaussian_cov",
        lf_mode, subject, model_name, data_mode, train_mode, "debug"
    )
else:
    result_path = path.join(
        project_root, "results", "dl_train_gaussian_cov",
        lf_mode, subject, model_name, data_mode, train_mode
    )

best_model_path = path.join(result_path, "best_model.pt")

os.makedirs(result_path, exist_ok=True)


# ============================================================
# INITIALIZATION
# ============================================================
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not path.isdir(syn_data_path):
    raise FileNotFoundError(syn_data_path)

if not path.isfile(lf_file):
    raise FileNotFoundError(lf_file)

lf = load_leadfield(lf_file)
leadfield, source_pos = lf["leadfield"], lf["pos"]
n_channels, n_sources = leadfield.shape[0], source_pos.shape[0]

run_folders = [f"gen_{idx}" for idx in range(1, 11)]
missing = [
    folder for folder in run_folders
    if not path.isdir(path.join(syn_data_path, folder))
]
if missing:
    raise FileNotFoundError(
        f"Generation folders tidak lengkap: {missing}\n"
        f"Dataset root: {syn_data_path}"
    )

if debug:
    run_folders = run_folders[:3]
if len(run_folders) < n_folds:
    raise ValueError(f"Generations ({len(run_folders)}) < folds ({n_folds})")

fold_splits = create_kfold_splits(
    run_folders, n_splits=n_folds, seed=seed
)

print("Device      :", device)
print("Synthetic   :", syn_data_path)
print("Leadfield   :", leadfield.shape)
print("Sources     :", source_pos.shape)
print("Generations :", run_folders)
print("Folds       :", len(fold_splits))
print("Results     :", result_path)


# ============================================================
# K-FOLD TRAINING
# ============================================================
best_overall = {
    "distance": float("inf"),
    "fold": -1,
    "epoch": -1,
}
fold_results = []
fold_histories = {}
training_start_time = time.time()
n_total_params = n_trainable_params = 0
model_size_mb = 0.0

for fold, (train_runs, val_runs) in enumerate(fold_splits, start=1):
    print(f"\n{'=' * 60}\nFOLD {fold}/{n_folds}")
    print("Train:", train_runs)
    print("Val  :", val_runs)
    print("=" * 60)

    # ============================================================
    # LAZY DATA INDEX
    # ============================================================

    train_samples = load_synthetic_index(
        syn_data_path,
        train_runs,
    )

    val_samples = load_synthetic_index(
        syn_data_path,
        val_runs,
    )

    train_loader, val_loader, train_dataset, val_dataset = (
        create_lazy_dataloaders(
            train_samples=train_samples,
            val_samples=val_samples,
            source_pos=source_pos,
            batch_size=batch_size,
            sigma_mm=gaussian_sigma_mm,
            radius_mm=gaussian_radius_mm,
            num_workers=4,
        )
    )

    n_times = train_dataset.n_times

    print(
        f"Train trials : {len(train_dataset):,}"
    )
    print(
        f"Val trials   : {len(val_dataset):,}"
    )
    print(
        f"Input shape  : "
        f"({n_channels}, {n_times})"
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = ESoLo(
        n_channels=n_channels,
        n_sources=n_sources,
        hidden_size=hidden_size,
        attention_size=attention_size,
        channel_reduction=channel_reduction,
        dropout=dropout,
    ).to(device)

    if fold == 1:
        summary(
            model,
            input_size=(
                1,
                n_channels,
                n_times,
            ),
            device=device.type,
        )

        n_total_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model_size_mb = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers())
        ) / 1024**2

        print("Total parameters     :", f"{n_total_params:,}")
        print("Trainable parameters :", f"{n_trainable_params:,}")
        print("Model size           :", f"{model_size_mb:.2f} MB")

    if loss_mode == "gaussian":
        criterion = GaussianSoftTargetLoss().to(device)
    elif loss_mode == "leadfield":
        criterion = GaussianLeadfieldLoss(
            leadfield=leadfield,
            classification_weight=classification_weight,
            leadfield_weight=leadfield_weight,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_fold = {
        "distance": float("inf"),
        "loss": float("inf"),
        "accuracy": 0.0,
        "epoch": 0,
        "checkpoint": None,
    }
    history = []
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            source_pos=source_pos,
            device=device,
        )

        epoch_time = time.time() - epoch_start_time

        train_loss = float(train_metrics["loss"])
        val_loss = float(val_metrics["loss"])
        val_distance = float(val_metrics["distance_mm"])
        val_accuracy = float(val_metrics["source_accuracy_percent"])

        values = {
            "train loss": train_loss,
            "validation loss": val_loss,
            "validation distance": val_distance,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise RuntimeError(
                    f"Fold {fold}, epoch {epoch}: {name} tidak valid: {value}"
                )

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        epoch_result = build_epoch_result(
            fold=fold,
            epoch=epoch,
            learning_rate=lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        epoch_result["epoch_time_sec"] = epoch_time
        history.append(epoch_result)

        print(
            f"Fold {fold}/{n_folds} | Epoch {epoch:03d}/{n_epochs}"
            f" | Train {train_loss:.4f} | Val {val_loss:.4f}"
            f" | Dist {val_distance:.2f} mm"
            f" | Acc {val_accuracy:.2f}%"
            f" | Time {epoch_time:.1f}s | LR {lr:.2e}"
        )

        if loss_mode == "leadfield":
            print(
                f"  Train components | "
                f"CE: {train_metrics['classification_loss']:.4f} | "
                f"Entropy: {train_metrics['target_entropy']:.4f} | "
                f"KL: {train_metrics['kl_divergence']:.4f} | "
                f"Leadfield: {train_metrics['leadfield_loss']:.4f} | "
                f"Similarity: {train_metrics['leadfield_similarity']:.4f}"
            )
            print(
                f"  Val components   | "
                f"CE: {val_metrics['classification_loss']:.4f} | "
                f"Entropy: {val_metrics['target_entropy']:.4f} | "
                f"KL: {val_metrics['kl_divergence']:.4f} | "
                f"Leadfield: {val_metrics['leadfield_loss']:.4f} | "
                f"Similarity: {val_metrics['leadfield_similarity']:.4f}"
            )
        else:
            print(
                f"  Classification   | "
                f"Train CE: {train_metrics['classification_loss']:.4f} | "
                f"Val CE: {val_metrics['classification_loss']:.4f}"
            )


        if val_distance < best_fold["distance"]:
            best_fold.update(
                distance=val_distance,
                loss=val_loss,
                accuracy=val_accuracy,
                epoch=epoch,
                checkpoint=build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    fold=fold,
                    epoch=epoch,
                    model_name=model_name,
                    data_mode=data_mode,
                    syn_data_path=syn_data_path,
                    train_runs=train_runs,
                    val_runs=val_runs,
                    source_pos=source_pos,
                    n_channels=n_channels,
                    n_times=n_times,
                    n_sources=n_sources,
                    hidden_size=hidden_size,
                    attention_size=attention_size,
                    channel_reduction=channel_reduction,
                    dropout=dropout,
                    gaussian_sigma_mm=gaussian_sigma_mm,
                    gaussian_radius_mm=gaussian_radius_mm,
                    loss_mode=loss_mode,
                    classification_weight=classification_weight,
                    leadfield_weight=leadfield_weight,
                    best_val_loss=val_loss,
                    best_val_distance_mm=val_distance,
                    best_val_source_accuracy_percent=val_accuracy,
                ),
            )
            no_improve = 0
            print(
                f"New best fold updated | Epoch {epoch}"
                f" | Val dist {val_distance:.2f} mm"
                f" | Val loss {val_loss:.4f}"
                f" | Val acc {val_accuracy:.2f}%"
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}"
                f" | Best epoch {best_fold['epoch']}"
            )
            break

    if best_fold["checkpoint"] is None:
        raise RuntimeError(f"Fold {fold}: tidak ada checkpoint valid.")

    fold_histories[f"Fold_{fold}"] = pd.DataFrame(history)
    fold_results.append({
        "fold": fold,
        "best_epoch": best_fold["epoch"],
        "best_val_loss": best_fold["loss"],
        "best_val_distance_mm": best_fold["distance"],
        "best_val_source_accuracy_percent": best_fold["accuracy"],
        "train_generations": ", ".join(train_runs),
        "val_generations": ", ".join(val_runs),
        "history_sheet": f"Fold_{fold}",
    })

    print(
        f"Fold {fold} completed"
        f" | Epoch {best_fold['epoch']}"
        f" | Loss {best_fold['loss']:.4f}"
        f" | Dist {best_fold['distance']:.2f} mm"
        f" | Acc {best_fold['accuracy']:.2f}%"
    )

    if best_fold["distance"] < best_overall["distance"]:
        best_overall.update(
            distance=best_fold["distance"],
            fold=fold,
            epoch=best_fold["epoch"],
        )
        torch.save(best_fold["checkpoint"], best_model_path)
        print(
            f"NEW BEST OVERALL | Fold {fold}"
            f" | Epoch {best_fold['epoch']}"
            f" | Dist {best_fold['distance']:.2f} mm"
        )

    del (
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        train_samples,
        val_samples,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# SAVE HISTORY AND PLOTS
# ============================================================
total_training_time = time.time() - training_start_time

training_info = pd.DataFrame([{
    "total_parameters": n_total_params,
    "trainable_parameters": n_trainable_params,
    "model_size_mb": model_size_mb,
    "total_training_time_sec": total_training_time,
}])

all_history_df = pd.concat(
    fold_histories.values(), ignore_index=True
)
fold_results_df = pd.DataFrame(fold_results)
history_summary = build_history_summary(all_history_df)

history_file = save_fold_outputs(
    result_path=result_path,
    fold_histories=fold_histories,
    all_history_df=all_history_df,
    history_summary=history_summary,
    fold_results_df=fold_results_df,
    training_info=training_info,
)
plot_mean_history(history_summary, result_path)


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'=' * 60}\nBEST OVERALL\n{'=' * 60}")
print("Fold     :", best_overall["fold"])
print("Epoch    :", best_overall["epoch"])
print("Val dist :", f"{best_overall['distance']:.2f} mm")
print("Model    :", best_model_path)
print("History  :", history_file)
print(
    f"Parameters: {n_total_params:,} total | "
    f"{n_trainable_params:,} trainable"
)
print(f"Model size: {model_size_mb:.2f} MB")
print(
    f"Training : {total_training_time:.1f} s | "
    f"{total_training_time / 60:.1f} min | "
    f"{total_training_time / 3600:.2f} h"
)
print()
print(fold_results_df.to_string(index=False))
