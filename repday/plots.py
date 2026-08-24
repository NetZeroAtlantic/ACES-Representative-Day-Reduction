from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_duration_curves(
    original_curves: Dict[str, np.ndarray],
    approx_curves: Dict[str, np.ndarray],
    output_dir: str,
    show: bool = False,
    save: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, original in original_curves.items():
        approx = approx_curves[name]
        x = np.linspace(0, 100, len(original))

        plt.figure(figsize=(10, 6))
        plt.plot(x, original, label=f"Original duration curve: {name}")
        plt.step(x, approx, where="post", label=f"Approximated duration curve: {name}")
        plt.xlabel("Duration [%]")
        plt.ylabel("Normalized value")
        plt.title(f"Original vs approximated duration curve - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(out / f"duration_curve_{name}.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
