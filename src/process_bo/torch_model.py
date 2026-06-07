"""PyTorch process-yield predictor trained on simulated noisy measurements."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from process_bo.objective import NoisyProcessObjective, ProcessBounds
from process_bo.sampling import latin_hypercube


class ProcessRegressor(nn.Module):
    """Small MLP for predicting process yield from process controls."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


@dataclass
class TrainingResult:
    rmse: float
    r2: float
    epochs: int
    train_samples: int
    test_samples: int
    model_path: Path | None
    metadata_path: Path | None


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def make_dataset(samples: int, *, seed: int, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    bounds = ProcessBounds.default()
    objective = NoisyProcessObjective(noise_std=noise_std, seed=seed, bounds=bounds)
    x = latin_hypercube(bounds.array, samples, seed=seed)
    y = objective.evaluate(x)
    return x, y


def train_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 11,
    epochs: int = 250,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    output_dir: str | Path | None = "artifacts",
) -> TrainingResult:
    _set_torch_seed(seed)
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train_scaled),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = ProcessRegressor(input_dim=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_test_scaled)).cpu().numpy()
    predictions = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    model_path = output_path / "process_regressor.pt" if output_path else None
    metadata_path = output_path / "process_regressor_metadata.json" if output_path else None
    if model_path and metadata_path:
        torch.save(model.state_dict(), model_path)
        metadata = {
            "rmse": rmse,
            "r2": r2,
            "epochs": epochs,
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "x_scaler_mean": x_scaler.mean_.tolist(),
            "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": y_scaler.mean_.tolist(),
            "y_scaler_scale": y_scaler.scale_.tolist(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        rmse=rmse,
        r2=r2,
        epochs=epochs,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        model_path=model_path,
        metadata_path=metadata_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--noise-std", type=float, default=1.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="artifacts")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    x, y = make_dataset(args.samples, seed=args.seed, noise_std=args.noise_std)
    result = train_regressor(
        x,
        y,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "rmse": result.rmse,
                "r2": result.r2,
                "epochs": result.epochs,
                "train_samples": result.train_samples,
                "test_samples": result.test_samples,
                "model_path": str(result.model_path) if result.model_path else None,
                "metadata_path": str(result.metadata_path) if result.metadata_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
