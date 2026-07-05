# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "numpy", "onnx", "onnxruntime"]
# ///
"""Train a QuickDraw sketch classifier and export it for the browser.

Run:  uv run train.py

Downloads a slice of Google's QuickDraw dataset (28x28 grayscale doodles),
trains a small CNN, and writes web/model.onnx + web/classes.json.
"""
import json
import pathlib
import urllib.request

import numpy as np
import torch
from torch import nn

CLASSES = ["airplane", "bicycle", "cat", "fish", "flower",
           "house", "pizza", "star", "tree", "umbrella"]
PER_CLASS = 12_000        # per category: 10k train + 2k test
TEST_PER_CLASS = 2_000
EPOCHS = 3
BATCH = 256
DATA_DIR = pathlib.Path(__file__).parent / "data"
WEB_DIR = pathlib.Path(__file__).parent / "web"
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def download(cls: str) -> np.ndarray:
    """Fetch the first PER_CLASS sketches of one category as (N, 28, 28) uint8."""
    cache = DATA_DIR / f"{cls}.npy"
    if cache.exists():
        return np.load(cache)
    # ponytail: full files are ~100MB each; an HTTP Range request grabs only
    # the first N rows (784 bytes/sketch). Drop the Range header if you ever
    # need the whole category.
    req = urllib.request.Request(URL.format(cls))
    req.add_header("Range", f"bytes=0-{256 + PER_CLASS * 784}")
    raw = urllib.request.urlopen(req).read()
    assert raw[:6] == b"\x93NUMPY" and raw[6] == 1, "unexpected .npy format"
    header_len = 10 + int.from_bytes(raw[8:10], "little")
    pixels = np.frombuffer(raw[header_len:], dtype=np.uint8)
    imgs = pixels[: PER_CLASS * 784].reshape(-1, 28, 28).copy()
    DATA_DIR.mkdir(exist_ok=True)
    np.save(cache, imgs)
    return imgs


def load_data():
    """Balanced train/test tensors, pixels scaled to [0, 1]."""
    train_x, test_x, train_y, test_y = [], [], [], []
    for label, cls in enumerate(CLASSES):
        imgs = download(cls)
        print(f"  {cls}: {len(imgs)} sketches")
        train_x.append(imgs[:-TEST_PER_CLASS])
        test_x.append(imgs[-TEST_PER_CLASS:])
        train_y += [label] * (len(imgs) - TEST_PER_CLASS)
        test_y += [label] * TEST_PER_CLASS

    def to_tensor(chunks):
        x = torch.tensor(np.concatenate(chunks), dtype=torch.float32)
        return x.div_(255).unsqueeze(1)  # (N, 1, 28, 28)

    return (to_tensor(train_x), torch.tensor(train_y),
            to_tensor(test_x), torch.tensor(test_y))


class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def accuracy(model, x, y, device):
    model.eval()
    correct = 0
    for i in range(0, len(x), BATCH):
        preds = model(x[i:i + BATCH].to(device)).argmax(1).cpu()
        correct += (preds == y[i:i + BATCH]).sum().item()
    return correct / len(x)


def main():
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Downloading data ({len(CLASSES)} classes x {PER_CLASS})...")
    x_train, y_train, x_test, y_test = load_data()

    model = Net(len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device} ({len(x_train)} sketches)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        loss_sum = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
        acc = accuracy(model, x_test, y_test, device)
        print(f"  epoch {epoch}: loss {loss_sum / len(perm):.4f}, test acc {acc:.1%}")

    # Export for the browser: fixed batch of 1 is all the web page needs.
    model.eval().cpu()
    WEB_DIR.mkdir(exist_ok=True)
    onnx_path = WEB_DIR / "model.onnx"
    torch.onnx.export(model, (torch.zeros(1, 1, 28, 28),), str(onnx_path),
                      input_names=["pixels"], output_names=["logits"],
                      dynamo=False)
    (WEB_DIR / "classes.json").write_text(json.dumps(CLASSES))

    # Self-check: onnxruntime must agree with PyTorch on real test sketches.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    sample = x_test[:1]
    ort_logits = sess.run(None, {"pixels": sample.numpy()})[0]
    assert isinstance(ort_logits, np.ndarray)  # narrow onnxruntime's loose return type
    torch_logits = model(sample).detach().numpy()
    assert np.allclose(ort_logits, torch_logits, atol=1e-3), "ONNX drifted from PyTorch"
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), ONNX matches PyTorch ✓")


if __name__ == "__main__":
    main()
