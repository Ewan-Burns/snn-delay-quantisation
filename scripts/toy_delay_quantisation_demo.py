"""
This code is for exploratory purposes only. It:
- Trains a tiny temporal pattern classifier with learnable per-class delays.
- Evaluates how simple post-training weight and delay quantisation changes accuracy and estimated parameter memory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 13
    num_classes: int = 6
    input_dim: int = 20
    train_size: int = 1200
    test_size: int = 600
    epochs: int = 220
    learning_rate: float = 0.03
    max_delay: float = 8.0
    reference_time: float = 13.0
    spike_noise: float = 0.85
    kernel_sigma: float = 0.7


class DelayClassifier(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight_logits = nn.Parameter(torch.randn(cfg.input_dim, cfg.num_classes) * 0.08)
        self.delay_logits = nn.Parameter(torch.randn(cfg.input_dim, cfg.num_classes) * 0.08)
        self.bias = nn.Parameter(torch.zeros(cfg.num_classes))

    def weights(self) -> torch.Tensor:
        return F.softplus(self.weight_logits)

    def delays(self) -> torch.Tensor:
        return self.cfg.max_delay * torch.sigmoid(self.delay_logits)

    def forward(
        self,
        spike_times: torch.Tensor,
        weights: torch.Tensor | None = None,
        delays: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self.weights() if weights is None else weights
        delays = self.delays() if delays is None else delays

        # Toy temporal-alignment mechanism: delays shift input spike times, and a Gaussian kernel rewards arrivals close to a class-independent reference time.
        # This is not intended as a biophysical neuron model.
        aligned = spike_times.unsqueeze(-1) + delays.unsqueeze(0)
        temporal_match = torch.exp(
            -0.5 * ((aligned - self.cfg.reference_time) / self.cfg.kernel_sigma) ** 2
        )
        return (temporal_match * weights.unsqueeze(0)).sum(dim=1) + self.bias


def make_data(cfg: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(cfg.seed)

    #Classes are defined by a latent spike-time template.
    # Samples are noisy observations of that template, making the task a minimal proxy for temporal coding.
    true_delays = rng.uniform(1.5, cfg.max_delay - 1.0, size=(cfg.num_classes, cfg.input_dim))
    class_centres = cfg.reference_time - true_delays

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = rng.integers(0, cfg.num_classes, size=n)
        times = class_centres[labels] + rng.normal(0.0, cfg.spike_noise, size=(n, cfg.input_dim))
        times = np.clip(times, 0.0, cfg.reference_time)
        return torch.tensor(times, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    train_x, train_y = sample(cfg.train_size)
    test_x, test_y = sample(cfg.test_size)
    return train_x, train_y, test_x, test_y


def quantize_uniform(values: torch.Tensor, bits: int, lo: float | None = None, hi: float | None = None) -> torch.Tensor:
    # Post-training uniform quantisation: this tests robustness of already
    # learned parameters, not quantisation-aware training during optimisation.
    if bits >= 32:
        return values.clone()
    levels = 2**bits
    lo_t = torch.tensor(float(values.min()) if lo is None else lo, dtype=values.dtype)
    hi_t = torch.tensor(float(values.max()) if hi is None else hi, dtype=values.dtype)
    if torch.isclose(hi_t, lo_t):
        return values.clone()
    scaled = (values - lo_t) / (hi_t - lo_t)
    quantized = torch.round(torch.clamp(scaled, 0.0, 1.0) * (levels - 1)) / (levels - 1)
    return quantized * (hi_t - lo_t) + lo_t


def quantize_ternary(values: torch.Tensor) -> torch.Tensor:
    # Simple ternary proxy for very low-precision weights. The bit count later
    # uses log2(3), which is an ideal storage estimate rather than a hardware
    # encoding guarantee.
    centred = values - values.mean()
    threshold = 0.5 * centred.abs().mean()
    scale = centred.abs().mean().clamp_min(1e-6)
    ternary = torch.where(
        centred > threshold,
        torch.ones_like(values),
        torch.where(centred < -threshold, -torch.ones_like(values), torch.zeros_like(values)),
    )
    return values.mean() + scale * ternary


@torch.no_grad()
def accuracy(model: DelayClassifier, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, delays: torch.Tensor) -> float:
    logits = model(x, weights=weights, delays=delays)
    return float((logits.argmax(dim=1) == y).float().mean().item())


def train_model(cfg: Config) -> tuple[DelayClassifier, torch.Tensor, torch.Tensor]:
    train_x, train_y, test_x, test_y = make_data(cfg)
    model = DelayClassifier(cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _ in range(cfg.epochs):
        optimiser.zero_grad()
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimiser.step()

    return model, test_x, test_y


def run(cfg: Config, output_dir: Path) -> list[dict[str, float | str]]:
    torch.manual_seed(cfg.seed)
    model, test_x, test_y = train_model(cfg)
    base_weights = model.weights().detach()
    base_delays = model.delays().detach()

    weight_options: list[tuple[str, float, torch.Tensor]] = [
        ("fp32", 32.0, base_weights),
        ("8-bit", 8.0, quantize_uniform(base_weights, 8)),
        ("4-bit", 4.0, quantize_uniform(base_weights, 4)),
        ("2-bit", 2.0, quantize_uniform(base_weights, 2)),
        ("ternary", float(np.log2(3)), quantize_ternary(base_weights)),
    ]
    delay_options: list[tuple[str, float, torch.Tensor]] = [
        ("fp32", 32.0, base_delays),
        ("4-bit", 4.0, quantize_uniform(base_delays, 4, lo=0.0, hi=cfg.max_delay)),
        ("3-bit", 3.0, quantize_uniform(base_delays, 3, lo=0.0, hi=cfg.max_delay)),
        ("2-bit", 2.0, quantize_uniform(base_delays, 2, lo=0.0, hi=cfg.max_delay)),
    ]

    rows: list[dict[str, float | str]] = []
    num_weights = cfg.input_dim * cfg.num_classes
    num_delays = cfg.input_dim * cfg.num_classes
    fp32_memory = num_weights * 32.0 + num_delays * 32.0

    for weight_name, weight_bits, weights in weight_options:
        for delay_name, delay_bits, delays in delay_options:
            # Parameter-memory estimate only. It excludes spike routing,
            # delay-buffer storage, runtime energy, and FPGA control overheads.
            memory_bits = num_weights * weight_bits + num_delays * delay_bits
            rows.append(
                {
                    "weight_precision": weight_name,
                    "delay_precision": delay_name,
                    "weight_bits": round(weight_bits, 3),
                    "delay_bits": round(delay_bits, 3),
                    "accuracy": round(accuracy(model, test_x, test_y, weights, delays), 4),
                    "memory_bits": round(memory_bits, 1),
                    "memory_ratio_vs_fp32": round(memory_bits / fp32_memory, 4),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "toy_delay_quantization_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "toy_delay_quantization_tradeoff.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for delay_name in ["fp32", "4-bit", "3-bit", "2-bit"]:
        series = [r for r in rows if r["delay_precision"] == delay_name]
        ax.plot(
            [float(r["memory_ratio_vs_fp32"]) for r in series],
            [float(r["accuracy"]) for r in series],
            marker="o",
            label=f"delay {delay_name}",
        )
    ax.set_xlabel("Estimated parameter memory vs fp32")
    ax.set_ylabel("Accuracy")
    ax.set_title("Toy learned-delay classifier: quantisation trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs)
    rows = run(cfg, args.output_dir)
    best = max(rows, key=lambda row: float(row["accuracy"]))
    print(f"Wrote results to: {args.output_dir}")
    print(f"Best accuracy: {best['accuracy']} with weights={best['weight_precision']}, delays={best['delay_precision']}")
    print("Note: this is an independent toy scaffold, not a DLTC reproduction.")


if __name__ == "__main__":
    main()
