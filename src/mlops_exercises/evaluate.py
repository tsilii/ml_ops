from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mlops_exercises.model import Model
from mlops_exercises.data import MyDataset


def evaluate(
    processed_dir: str = "data/processed",
    model_path: str = "models/model.pt",
    batch_size: int = 256,
) -> None:
    """
    Evaluate a trained model on the test split and print classification accuracy.

    Parameters
    ----------
    processed_dir : str
        Directory containing processed tensors (test_images.pt, test_labels.pt).
    model_path : str
        Path to the saved model weights (PyTorch state_dict).
    batch_size : int
        Batch size used for evaluation. Larger values are usually faster on GPU.

    Returns
    -------
    None
        This function prints the accuracy to stdout.

    Raises
    ------
    FileNotFoundError
        If model_path or the required processed data files are missing.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = MyDataset(processed_dir=processed_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).long()
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
