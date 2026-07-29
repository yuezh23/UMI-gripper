"""
Train LSTM/MLP models for force-state classification.

Modalities:
    force:
        Temporal force sequence
        Shape: (N, T, 12)

    gelsight:
        Aggregated GelSight features
        Shape: (N, G)

    combined:
        Temporal force + aggregated GelSight
        Force:     (N, T, 12)
        GelSight:  (N, G)

Splitting:
    stratified_episode:
        Split by episode_id, stratified by
        object_type + label.

        Train / validation / test are mutually exclusive.

    leave_one_object_out:
        One object is completely held out for testing.

Metrics:
    accuracy
    macro-F1
    confusion matrix
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# ============================================================
# Labels
# ============================================================

LABELS = (
    "too_low",
    "fine",
    "too_high",
)

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "artifacts/force_state/"
            "lstm_force_gelsight_v3.npz"
        ),
    )

    parser.add_argument(
        "--split",
        choices=(
            "stratified_episode",
            "leave_one_object_out",
        ),
        default="stratified_episode",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/force_state/"
            "lstm_models"
        ),
    )

    parser.add_argument(
        "--modalities",
        type=str,
        default="force,gelsight,combined",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Reproducibility.
    # Slightly slower, but appropriate for experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class ForceGelSightDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        force: np.ndarray,
        gelsight: np.ndarray,
        labels: np.ndarray,
    ):

        self.force = torch.from_numpy(
            force.astype(np.float32)
        )

        self.gelsight = torch.from_numpy(
            gelsight.astype(np.float32)
        )

        self.labels = torch.from_numpy(
            labels.astype(np.int64)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.force[index],
            self.gelsight[index],
            self.labels[index],
        )


# ============================================================
# Models
# ============================================================

class ForceLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):

        super().__init__()

        lstm_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                len(LABELS),
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        output, _ = self.lstm(x)

        # Last timestep
        last = output[:, -1, :]

        return self.classifier(last)


class GelSightMLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        dropout: float,
    ):

        super().__init__()

        hidden1 = min(
            256,
            max(64, input_size),
        )

        hidden2 = 64

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden1,
                hidden2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden2,
                len(LABELS),
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        return self.network(x)


class CombinedModel(nn.Module):

    def __init__(
        self,
        force_input_size: int,
        gelsight_input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):

        super().__init__()

        lstm_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.force_lstm = nn.LSTM(
            input_size=force_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        gelsight_hidden = min(
            256,
            max(64, gelsight_input_size),
        )

        self.gelsight_mlp = nn.Sequential(
            nn.Linear(
                gelsight_input_size,
                gelsight_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                gelsight_hidden,
                hidden_size,
            ),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                len(LABELS),
            ),
        )

    def forward(
        self,
        force: torch.Tensor,
        gelsight: torch.Tensor,
    ):

        force_output, _ = self.force_lstm(
            force
        )

        force_embedding = (
            force_output[:, -1, :]
        )

        gelsight_embedding = (
            self.gelsight_mlp(gelsight)
        )

        combined = torch.cat(
            [
                force_embedding,
                gelsight_embedding,
            ],
            dim=1,
        )

        return self.classifier(
            combined
        )


# ============================================================
# Episode-level split
# ============================================================

def split_episode_ids(
    episode_ids: np.ndarray,
    object_types: np.ndarray,
    labels: np.ndarray,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[
    set[str],
    set[str],
    set[str],
]:
    """
    Split by episode_id.

    Stratification is performed using:

        object_type + label

    This guarantees that samples from the same
    episode never appear in different splits.
    """

    if not (
        0.0 <= val_fraction < 1.0
    ):
        raise ValueError(
            "val_fraction must be in [0, 1)."
        )

    if not (
        0.0 <= test_fraction < 1.0
    ):
        raise ValueError(
            "test_fraction must be in [0, 1)."
        )

    if (
        val_fraction + test_fraction
        >= 1.0
    ):
        raise ValueError(
            "val_fraction + test_fraction "
            "must be < 1."
        )

    # --------------------------------------------------------
    # One row per episode
    # --------------------------------------------------------

    records = {}

    for episode_id, object_type, label in zip(
        episode_ids,
        object_types,
        labels,
    ):

        episode_id = str(
            episode_id
        )

        if episode_id in records:
            continue

        records[episode_id] = (
            str(object_type),
            int(label),
        )

    # --------------------------------------------------------
    # Group episode IDs by object + label
    # --------------------------------------------------------

    groups: dict[
        tuple[str, int],
        list[str],
    ] = {}

    for episode_id, (
        object_type,
        label,
    ) in records.items():

        key = (
            object_type,
            label,
        )

        groups.setdefault(
            key,
            [],
        ).append(
            episode_id
        )

    rng = np.random.default_rng(
        seed
    )

    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    # --------------------------------------------------------
    # Split each object + label group
    # --------------------------------------------------------

    for key in sorted(groups):

        ids = np.asarray(
            groups[key],
            dtype=str,
        )

        rng.shuffle(ids)

        n = len(ids)

        # ----------------------------------------------------
        # Test
        #
        # Important:
        # If a group has >= 3 episodes and test_fraction > 0,
        # always reserve at least one test episode.
        # ----------------------------------------------------

        if (
            test_fraction > 0
            and n >= 3
        ):

            n_test = max(
                1,
                int(
                    round(
                        n
                        * test_fraction
                    )
                ),
            )

        else:

            n_test = 0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        remaining = n - n_test

        if (
            val_fraction > 0
            and remaining >= 3
        ):

            n_val = max(
                1,
                int(
                    round(
                        n
                        * val_fraction
                    )
                ),
            )

            # Leave at least one training episode.
            n_val = min(
                n_val,
                remaining - 1,
            )

        else:

            n_val = 0

        test_part = ids[
            :n_test
        ]

        val_part = ids[
            n_test:
            n_test + n_val
        ]

        train_part = ids[
            n_test + n_val:
        ]

        test_ids.update(
            test_part.tolist()
        )

        val_ids.update(
            val_part.tolist()
        )

        train_ids.update(
            train_part.tolist()
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if (
        train_ids
        & val_ids
    ):
        raise RuntimeError(
            "Train/validation episode overlap."
        )

    if (
        train_ids
        & test_ids
    ):
        raise RuntimeError(
            "Train/test episode overlap."
        )

    if (
        val_ids
        & test_ids
    ):
        raise RuntimeError(
            "Validation/test episode overlap."
        )

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# Folds
# ============================================================

def folds(
    episode_ids: np.ndarray,
    object_types: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
):
    """
    Generate train/val/test episode IDs.
    """

    if args.split == "stratified_episode":

        train_ids, val_ids, test_ids = (
            split_episode_ids(
                episode_ids=episode_ids,
                object_types=object_types,
                labels=labels,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
        )

        yield (
            "within_object",
            train_ids,
            val_ids,
            test_ids,
        )

        return

    # ========================================================
    # Leave-One-Object-Out
    # ========================================================

    unique_objects = sorted(
        set(
            object_types.astype(str)
        )
    )

    for index, object_name in enumerate(
        unique_objects
    ):

        test_mask = (
            object_types.astype(str)
            == object_name
        )

        test_ids = set(
            episode_ids[
                test_mask
            ].astype(str)
        )

        remaining_mask = (
            ~test_mask
        )

        remaining_episode_ids = (
            episode_ids[
                remaining_mask
            ]
        )

        remaining_objects = (
            object_types[
                remaining_mask
            ]
        )

        remaining_labels = (
            labels[
                remaining_mask
            ]
        )

        train_ids, val_ids, _ = (
            split_episode_ids(
                episode_ids=(
                    remaining_episode_ids
                ),
                object_types=(
                    remaining_objects
                ),
                labels=(
                    remaining_labels
                ),
                seed=args.seed + index,
                val_fraction=args.val_fraction,
                test_fraction=0.0,
            )
        )

        yield (
            f"test_{object_name}",
            train_ids,
            val_ids,
            test_ids,
        )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    if len(y_true) == 0:

        raise RuntimeError(
            "Cannot calculate metrics: "
            "test set is empty."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(
            len(LABELS)
        ),
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    modality,
):

    model.eval()

    all_y = []
    all_pred = []

    for (
        force,
        gelsight,
        labels,
    ) in loader:

        force = force.to(
            device
        )

        gelsight = gelsight.to(
            device
        )

        if modality == "force":

            logits = model(
                force
            )

        elif modality == "gelsight":

            logits = model(
                gelsight
            )

        else:

            logits = model(
                force,
                gelsight,
            )

        pred = (
            logits
            .argmax(dim=1)
        )

        all_y.extend(
            labels.cpu().numpy()
        )

        all_pred.extend(
            pred.cpu().numpy()
        )

    y_true = np.asarray(
        all_y,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        all_pred,
        dtype=np.int64,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            len(LABELS)
        ),
    )

    return (
        metrics,
        cm,
        y_true,
        y_pred,
    )


# ============================================================
# Create model
# ============================================================

def create_model(
    modality: str,
    force_dim: int,
    gelsight_dim: int,
    args: argparse.Namespace,
):

    if modality == "force":

        return ForceLSTM(
            input_size=force_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    if modality == "gelsight":

        return GelSightMLP(
            input_size=gelsight_dim,
            dropout=args.dropout,
        )

    if modality == "combined":

        return CombinedModel(
            force_input_size=force_dim,
            gelsight_input_size=gelsight_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    raise ValueError(
        f"Unknown modality: {modality}"
    )


# ============================================================
# Train one fold
# ============================================================

def train_fold(
    force: np.ndarray,
    gelsight: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    modality: str,
    fold_name: str,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
):

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    episode_ids = (
        episode_ids.astype(str)
    )

    train_mask = np.isin(
        episode_ids,
        list(train_ids),
    )

    val_mask = np.isin(
        episode_ids,
        list(val_ids),
    )

    test_mask = np.isin(
        episode_ids,
        list(test_ids),
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not train_mask.any():

        raise RuntimeError(
            f"{fold_name}: empty train set."
        )

    if not val_mask.any():

        raise RuntimeError(
            f"{fold_name}: empty validation set."
        )

    if not test_mask.any():

        raise RuntimeError(
            f"{fold_name}: empty test set."
        )

    # --------------------------------------------------------
    # Extract data
    # --------------------------------------------------------

    train_force = force[
        train_mask
    ]

    val_force = force[
        val_mask
    ]

    test_force = force[
        test_mask
    ]

    train_gelsight = gelsight[
        train_mask
    ]

    val_gelsight = gelsight[
        val_mask
    ]

    test_gelsight = gelsight[
        test_mask
    ]

    train_labels = labels[
        train_mask
    ]

    val_labels = labels[
        val_mask
    ]

    test_labels = labels[
        test_mask
    ]

    # --------------------------------------------------------
    # Print split
    # --------------------------------------------------------

    print(
        f"Train: "
        f"{train_force.shape}"
    )

    print(
        f"Val:   "
        f"{val_force.shape}"
    )

    print(
        f"Test:  "
        f"{test_force.shape}"
    )

    print(
        f"Episodes: "
        f"train={len(train_ids)}, "
        f"val={len(val_ids)}, "
        f"test={len(test_ids)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        ForceGelSightDataset(
            train_force,
            train_gelsight,
            train_labels,
        )
    )

    val_dataset = (
        ForceGelSightDataset(
            val_force,
            val_gelsight,
            val_labels,
        )
    )

    test_dataset = (
        ForceGelSightDataset(
            test_force,
            test_gelsight,
            test_labels,
        )
    )

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
    )

    val_loader = (
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )

    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    force_dim = (
        force.shape[-1]
    )

    gelsight_dim = (
        gelsight.shape[-1]
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        modality=modality,
        force_dim=force_dim,
        gelsight_dim=gelsight_dim,
        args=args,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_state = None
    best_val_f1 = -np.inf
    best_epoch = 0
    patience_counter = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        losses = []

        for (
            force_batch,
            gelsight_batch,
            labels_batch,
        ) in train_loader:

            force_batch = (
                force_batch.to(device)
            )

            gelsight_batch = (
                gelsight_batch.to(device)
            )

            labels_batch = (
                labels_batch.to(device)
            )

            optimizer.zero_grad()

            if modality == "force":

                logits = model(
                    force_batch
                )

            elif modality == "gelsight":

                logits = model(
                    gelsight_batch
                )

            else:

                logits = model(
                    force_batch,
                    gelsight_batch,
                )

            loss = criterion(
                logits,
                labels_batch,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _, _ = (
            evaluate(
                model=model,
                loader=val_loader,
                device=device,
                modality=modality,
            )
        )

        train_loss = (
            float(
                np.mean(losses)
            )
            if losses
            else float("nan")
        )

        val_f1 = (
            val_metrics["macro_f1"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc="
            f"{val_metrics['accuracy']:.4f} | "
            f"val_macro_f1="
            f"{val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= args.patience
        ):

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No best model state was saved."
        )

    model.load_state_dict(
        best_state
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    train_metrics, _, _, _ = (
        evaluate(
            model=model,
            loader=train_loader,
            device=device,
            modality=modality,
        )
    )

    val_metrics, _, _, _ = (
        evaluate(
            model=model,
            loader=val_loader,
            device=device,
            modality=modality,
        )
    )

    test_metrics, cm, y_true, y_pred = (
        evaluate(
            model=model,
            loader=test_loader,
            device=device,
            modality=modality,
        )
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print()

    print(
        f"{modality}: "
        f"macro_f1="
        f"{test_metrics['macro_f1']:.4f}, "
        f"train="
        f"{train_metrics['macro_f1']:.3f}, "
        f"val="
        f"{val_metrics['macro_f1']:.3f}, "
        f"test="
        f"{test_metrics['macro_f1']:.3f}"
    )

    print()

    print(
        "Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        "          "
        + "  ".join(
            f"{label:>9}"
            for label in LABELS
        )
    )

    for i, label in enumerate(
        LABELS
    ):

        print(
            f"{label:>9} "
            + "  ".join(
                f"{int(x):9d}"
                for x in cm[i]
            )
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    fold_output = (
        output_dir
        / fold_name
        / modality
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        fold_output
        / "model.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "modality":
                modality,

            "force_dim":
                force_dim,

            "gelsight_dim":
                gelsight_dim,

            "hidden_size":
                args.hidden_size,

            "num_layers":
                args.num_layers,

            "dropout":
                args.dropout,

            "labels":
                LABELS,

            "best_epoch":
                best_epoch,
        },
        model_path,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "fold": fold_name,
        "modality": modality,

        "train_samples":
            int(train_mask.sum()),

        "val_samples":
            int(val_mask.sum()),

        "test_samples":
            int(test_mask.sum()),

        "train_episodes":
            len(train_ids),

        "val_episodes":
            len(val_ids),

        "test_episodes":
            len(test_ids),

        "best_epoch":
            best_epoch,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_macro_f1":
            train_metrics["macro_f1"],

        "val_accuracy":
            val_metrics["accuracy"],

        "val_macro_f1":
            val_metrics["macro_f1"],

        "test_accuracy":
            test_metrics["accuracy"],

        "test_macro_f1":
            test_metrics["macro_f1"],

        "confusion_matrix":
            cm.tolist(),
    }

    with (
        fold_output
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metrics,
            handle,
            indent=2,
        )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not args.dataset.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{args.dataset}"
        )

    data = np.load(
        args.dataset,
        allow_pickle=True,
    )

    force = data[
        "X_force"
    ].astype(
        np.float32
    )

    gelsight = data[
        "X_gelsight"
    ].astype(
        np.float32
    )

    labels = data[
        "y"
    ].astype(
        np.int64
    )

    episode_ids = data[
        "episode_ids"
    ].astype(str)

    object_types = data[
        "object_types"
    ].astype(str)

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    N = len(labels)

    if force.shape[0] != N:

        raise ValueError(
            "X_force and y have "
            "different number of samples."
        )

    if gelsight.shape[0] != N:

        raise ValueError(
            "X_gelsight and y have "
            "different number of samples."
        )

    if len(episode_ids) != N:

        raise ValueError(
            "episode_ids and y have "
            "different number of samples."
        )

    if len(object_types) != N:

        raise ValueError(
            "object_types and y have "
            "different number of samples."
        )

    # --------------------------------------------------------
    # Print dataset information
    # --------------------------------------------------------

    print(
        f"Force shape: "
        f"{force.shape}"
    )

    print(
        f"GelSight shape: "
        f"{gelsight.shape}"
    )

    print(
        f"Samples: "
        f"{N}"
    )

    print(
        f"Episodes: "
        f"{len(set(episode_ids))}"
    )

    print(
        f"Objects: "
        f"{len(set(object_types))}"
    )

    print()

    print(
        "Label distribution:"
    )

    for index, label in enumerate(
        LABELS
    ):

        print(
            f"  {label}: "
            f"{int((labels == index).sum())}"
        )

    print()

    # --------------------------------------------------------
    # Modalities
    # --------------------------------------------------------

    modalities = [
        x.strip()
        for x in args.modalities.split(",")
        if x.strip()
    ]

    valid_modalities = {
        "force",
        "gelsight",
        "combined",
    }

    invalid = (
        set(modalities)
        - valid_modalities
    )

    if invalid:

        raise ValueError(
            f"Unknown modalities: "
            f"{sorted(invalid)}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    all_results = []

    for (
        fold_name,
        train_ids,
        val_ids,
        test_ids,
    ) in folds(
        episode_ids=episode_ids,
        object_types=object_types,
        labels=labels,
        args=args,
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"Fold: {fold_name}"
        )

        print(
            "=" * 70
        )

        print(
            f"Train episodes: "
            f"{len(train_ids)}"
        )

        print(
            f"Val episodes: "
            f"{len(val_ids)}"
        )

        print(
            f"Test episodes: "
            f"{len(test_ids)}"
        )

        # ----------------------------------------------------
        # Important safety check
        # ----------------------------------------------------

        if not test_ids:

            raise RuntimeError(
                f"Fold {fold_name} has "
                "ZERO test episodes. "
                "Check split configuration."
            )

        for modality in modalities:

            print()

            print(
                "-" * 70
            )

            print(
                f"Training modality: "
                f"{modality}"
            )

            print(
                "-" * 70
            )

            result = train_fold(
                force=force,
                gelsight=gelsight,
                labels=labels,
                episode_ids=episode_ids,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                modality=modality,
                fold_name=fold_name,
                output_dir=args.output_dir,
                args=args,
                device=device,
            )

            all_results.append(
                result
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Final summary"
    )

    print(
        "=" * 70
    )

    for result in all_results:

        print(
            f"{result['fold']:>25} | "
            f"{result['modality']:>10} | "
            f"accuracy="
            f"{result['test_accuracy']:.4f} | "
            f"macro_f1="
            f"{result['test_macro_f1']:.4f}"
        )

    print()


if __name__ == "__main__":
    main()