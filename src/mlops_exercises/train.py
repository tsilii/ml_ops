"""
train.py

Trains a small Convolutional Neural Network (CNN) on the processed (normalized) MNIST dataset.

Key ideas for someone new to ML:
- We load preprocessed tensors from data/processed (created by data.py).
- We split the training data into "train" and "validation" sets.
  * Train set: used to update the model parameters.
  * Validation set: used only to monitor performance and detect overfitting.
- We optimize the model using gradient descent (Adam) to minimize a loss function.
- We save:
  1) The trained model weights to models/model.pt
  2) A training curve figure to reports/figures/training_curve.png
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from mlops_exercises.model import Model
from mlops_exercises.data import MyDataset


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute accuracy for a batch.

    logits: raw model outputs of shape [batch_size, num_classes]
            (higher value = more confident for that class)
    y:      true labels of shape [batch_size] with integers 0..9

    We pick the class with the largest logit and compare to y.
    """
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train(
    processed_dir: str = "data/processed",
    model_out: str = "models/model.pt",
    fig_out: str = "reports/figures/training_curve.png",
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Main training function.

    Parameters:
    - processed_dir: folder with processed tensors (train_images.pt, train_labels.pt, ...)
    - model_out: where to save trained weights (state_dict)
    - fig_out: where to save a loss curve plot
    - batch_size: number of examples per gradient update
    - lr: learning rate (step size) for the optimizer
    - epochs: number of passes over the training set
    - val_fraction: fraction of training data held out for validation
    - seed: random seed for reproducibility of the split
    """
    # Setting a random seed helps you reproduce the same train/val split and (partially) the same training.
    torch.manual_seed(seed)

    # Use GPU if available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # 1) Load data
    # -------------------------
    # MyDataset loads tensors from data/processed by default:
    # - train_images.pt and train_labels.pt for split="train"
    full_train = MyDataset(processed_dir=processed_dir, split="train")

    # Split the full training set into train/validation subsets.
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    # DataLoader handles batching and shuffling.
    # shuffle=True for training improves learning because batches change each epoch.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # shuffle=False for validation ensures deterministic evaluation.
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 2) Create model + training components
    # -------------------------
    model = Model().to(device)
    print(model)

    # CrossEntropyLoss is standard for multi-class classification with integer labels.
    criterion = nn.CrossEntropyLoss()

    # Adam is a popular optimizer that usually works well out-of-the-box.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # We will store metrics for plotting after training.
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # -------------------------
    # 3) Training loop
    # -------------------------
    for epoch in range(1, epochs + 1):
        # Put model in training mode (enables behaviors like dropout/batchnorm if present).
        model.train()

        # We'll accumulate totals so we can compute the average loss/accuracy over the epoch.
        running_loss = 0.0
        running_correct = 0.0  # number of correct predictions (not yet divided by count)

        for x, y in train_loader:
            # Move batch to CPU/GPU.
            x = x.to(device)
            y = y.to(device).long()  # labels must be integer class indices for CrossEntropyLoss

            # Reset gradients from previous step.
            optimizer.zero_grad()

            # Forward pass: compute model predictions.
            logits = model(x)

            # Loss: how wrong the model is (lower is better).
            loss = criterion(logits, y)

            # Backward pass: compute gradients of loss w.r.t. model parameters.
            loss.backward()

            # Update parameters using the optimizer.
            optimizer.step()

            # Multiply by batch size so we can compute correct epoch average later.
            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).float().sum().item()

        epoch_train_loss = running_loss / n_train
        epoch_train_acc = running_correct / n_train

        # -------------------------
        # 4) Validation loop (no parameter updates)
        # -------------------------
        model.eval()  # evaluation mode
        vloss = 0.0
        vcorrect = 0.0

        # no_grad disables gradient tracking -> faster and less memory usage.
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).long()

                logits = model(x)
                loss = criterion(logits, y)

                vloss += loss.item() * x.size(0)
                vcorrect += (logits.argmax(dim=1) == y).float().sum().item()

        # If val_fraction is 0, n_val becomes 0; guard against division by zero.
        epoch_val_loss = vloss / n_val if n_val > 0 else float("nan")
        epoch_val_acc = vcorrect / n_val if n_val > 0 else float("nan")

        # Store metrics for plotting.
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {epoch_train_loss:.4f} acc {epoch_train_acc:.4f} | "
            f"val loss {epoch_val_loss:.4f} acc {epoch_val_acc:.4f}"
        )

    # -------------------------
    # 5) Save artifacts (model + figure)
    # -------------------------
    # Save only weights (state_dict). This is the most common PyTorch practice.
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Saved model to: {model_out}")

    # Plot training/validation loss curves.
    Path(fig_out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_out)
    print(f"Saved training curve to: {fig_out}")


if __name__ == "__main__":
    train()
    # Running th
