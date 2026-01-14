from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from mlops_exercises.model import Model
from mlops_exercises.data import MyDataset


def extract_embeddings(model: Model, x: torch.Tensor) -> torch.Tensor:
    """
    Extract intermediate feature vectors (embeddings) from the CNN for a batch of images.

    The embedding is taken from the classifier just before the final classification layer.
    For this specific Model implementation, it corresponds to the 128-dimensional vector
    after the first Linear layer and ReLU.

    Parameters
    ----------
    model : Model
        A trained CNN model with `features` and `classifier` modules.
    x : torch.Tensor
        Input images of shape [N, 1, 28, 28].

    Returns
    -------
    torch.Tensor
        Embeddings of shape [N, 128] on the same device as `x`.
    """
    feats = model.features(x)  # [N, 64, 7, 7]
    feats = feats.flatten(1)  # [N, 3136]
    h = model.classifier[1](feats)  # Linear -> [N, 128]
    h = model.classifier[2](h)  # ReLU  -> [N, 128]
    return h


def main(
    processed_dir: Path,
    model_path: Path,
    out_path: Path,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> None:
    """
    Create a t-SNE plot of CNN embeddings for the training set and save it to disk.

    Workflow:
    1) Load a pretrained model from `model_path`.
    2) Load training data from `processed_dir`.
    3) Extract 128-d embeddings for up to `max_samples` images.
    4) Run t-SNE to reduce embeddings to 2D.
    5) Save a scatter plot to `out_path`.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed training tensors (train_images.pt, train_labels.pt).
    model_path : Path
        Path to trained model weights (state_dict).
    out_path : Path
        File path where the resulting plot is saved (e.g., reports/figures/tsne.png).
    max_samples : int
        Maximum number of training samples to embed (t-SNE is slow; keep this moderate).
    batch_size : int
        Batch size used for embedding extraction.
    seed : int
        Random seed for reproducibility of sampling and t-SNE initialization.

    Returns
    -------
    None
        Saves a plot and prints status information.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = Model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load TRAIN data
    train_ds = MyDataset(processed_dir=processed_dir, split="train")
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # Collect embeddings + labels (optionally sub-sample)
    all_embeds = []
    all_labels = []
    collected = 0

    with torch.no_grad():
        for x, y in loader:
            if collected >= max_samples:
                break

            # Trim last batch if it would exceed max_samples
            remaining = max_samples - collected
            if x.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]

            x = x.to(device)
            emb = extract_embeddings(model, x).cpu()  # [B, 128]

            all_embeds.append(emb)
            all_labels.append(y.cpu())
            collected += x.size(0)

    embeds = torch.cat(all_embeds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    print(f"Collected embeddings: {embeds.shape} (N, 128)")

    # t-SNE to 2D
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    z = tsne.fit_transform(embeds)  # [N, 2]

    # Plot
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, s=8)
    plt.title("t-SNE of CNN embeddings (train set)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(scatter, label="digit class")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-path", type=Path, default=Path("models/model.pt"))
    parser.add_argument("--out-path", type=Path, default=Path("reports/figures/tsne.png"))
    parser.add_argument("--max-samples", type=int, default=5000, help="t-SNE is slow; 5k is a reasonable default.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        processed_dir=args.processed_dir,
        model_path=args.model_path,
        out_path=args.out_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
