import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(results: dict, cfg, output_path: str = "learning_curves.png"):
    """Plot average learning trajectories comparing convergence speed across strategies."""
    plt.figure(figsize=(9, 6))

    for name, losses in results.items():
        # Extracted loss curves array shape: (n_seeds, T + 1)
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        steps = np.arange(len(mean_loss))

        plt.plot(steps, mean_loss, label=name, marker='o', linewidth=2)
        plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15)

    plt.title(f"Active Learning Convergence (T={cfg.al.T})", fontsize=14, pad=10)
    plt.xlabel("Number of Acquisitions", fontsize=12)
    plt.ylabel("Validation MSE Loss", fontsize=12)
    plt.yscale("log")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)

    full_path = os.path.join(cfg.results_dir, output_path)
    plt.savefig(full_path, dpi=300)
    plt.close()
    print(f"Chart successfully saved to {full_path}")