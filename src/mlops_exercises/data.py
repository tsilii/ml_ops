from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    Dataset wrapper around the *processed* MNIST tensors.

    This class expects files created by the preprocessing step:
        python src/mlops_exercises/data.py data/raw data/processed

    Expected files inside processed_dir:
      - train_images.pt, train_labels.pt
      - test_images.pt,  test_labels.pt
    """

    def __init__(self, processed_dir: str | Path = "data/processed", split: str = "train") -> None:
        super().__init__()
        processed_dir = Path(processed_dir)

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        if split == "train":
            self.images = torch.load(processed_dir / "train_images.pt")
            self.labels = torch.load(processed_dir / "train_labels.pt")
        else:
            self.images = torch.load(processed_dir / "test_images.pt")
            self.labels = torch.load(processed_dir / "test_labels.pt")

        if len(self.images) != len(self.labels):
            raise ValueError(f"Mismatched lengths: images={len(self.images)} labels={len(self.labels)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def load_and_concat(pattern: str, folder: Path) -> torch.Tensor:
    """Load multiple tensors matching pattern in folder and concatenate along dim=0."""
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {folder}")
    parts = [torch.load(f) for f in files]
    return torch.cat(parts, dim=0)


def ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure images are [N, 1, 28, 28]. Accepts [N, 28, 28] or [N, 1, 28, 28]."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor for images, got shape {tuple(x.shape)}")
    return x


def main(raw_dir: Path, processed_dir: Path) -> None:
    """
    Preprocess corrupt MNIST:
    - Load chunked train tensors and concatenate
    - Load test tensors
    - Normalize using TRAIN mean/std
    - Save processed tensors to processed_dir
    """
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    ds_dir = raw_dir / "corruptmnist"
    if not ds_dir.exists():
        raise FileNotFoundError(f"Expected folder {ds_dir}. Did you place corruptmnist under data/raw/?")

    # Train set is chunked
    train_images = load_and_concat("train_images_*.pt", ds_dir)
    train_labels = load_and_concat("train_target_*.pt", ds_dir)

    # Test set is single files
    test_images = torch.load(ds_dir / "test_images.pt")
    test_labels = torch.load(ds_dir / "test_target.pt")

    train_images = ensure_nchw(train_images).float()
    test_images = ensure_nchw(test_images).float()

    # Normalize using TRAIN statistics
    mean = train_images.mean()
    std = train_images.std(unbiased=False)

    train_images = (train_images - mean) / (std + 1e-8)
    test_images = (test_images - mean) / (std + 1e-8)

    # Save processed tensors with clean names
    torch.save(train_images, processed_dir / "train_images.pt")
    torch.save(train_labels, processed_dir / "train_labels.pt")
    torch.save(test_images, processed_dir / "test_images.pt")
    torch.save(test_labels, processed_dir / "test_labels.pt")
    torch.save({"mean": mean, "std": std}, processed_dir / "stats.pt")

    print("Preprocessing complete.")
    print(f"Train images: {tuple(train_images.shape)}  Train labels: {tuple(train_labels.shape)}")
    print(f"Test  images: {tuple(test_images.shape)}   Test  labels: {tuple(test_labels.shape)}")
    print(f"Train mean: {mean.item():.6f}, Train std: {std.item():.6f}")
    print(f"Saved to: {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corrupt MNIST into normalized tensors.")
    parser.add_argument("raw_dir", type=Path, help="Raw data dir, e.g. data/raw")
    parser.add_argument("processed_dir", type=Path, help="Output dir, e.g. data/processed")
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir)
