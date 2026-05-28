import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT = os.getcwd()
CSV_PATH = "training_metrics.csv"
CHECKPOINT_PATH = os.path.join(ROOT, "single system", CSV_PATH)
SMOOTHING_WINDOW = 50  # Rolling average window size default 50


def smooth_data(series, window=SMOOTHING_WINDOW):
    """Apply rolling average smoothing to reduce noise."""
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed


def main():
    df = pd.read_csv(CHECKPOINT_PATH)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Reward
    smoothed = smooth_data(df["ep_rew_mean"])
    axes[0, 0].plot(df["timesteps"], smoothed, linewidth=2)
    axes[0, 0].set_title("Episode Reward Mean")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].grid(True, alpha=0.3)

    # Episode length
    smoothed = smooth_data(df["ep_len_mean"])
    axes[0, 1].plot(df["timesteps"], smoothed, linewidth=2)
    axes[0, 1].set_title("Episode Length Mean")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].grid(True, alpha=0.3)

    # KL
    smoothed = smooth_data(df["approx_kl"])
    axes[1, 0].plot(df["timesteps"], smoothed, linewidth=2)
    axes[1, 0].set_title("Approx KL")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].grid(True, alpha=0.3)

    # Explained variance
    smoothed = smooth_data(df["explained_variance"])
    axes[1, 1].plot(df["timesteps"], smoothed, linewidth=2)
    axes[1, 1].set_title("Explained Variance")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].grid(True, alpha=0.3)

    # Value loss
    smoothed = smooth_data(df["value_loss"])
    axes[2, 0].plot(df["timesteps"], smoothed, linewidth=2)
    axes[2, 0].set_title("Value Loss")
    axes[2, 0].set_xlabel("Timesteps")
    axes[2, 0].grid(True, alpha=0.3)

    # Entropy
    smoothed = smooth_data(df["entropy_loss"])
    axes[2, 1].plot(df["timesteps"], smoothed, linewidth=2)
    axes[2, 1].set_title("Entropy Loss")
    axes[2, 1].set_xlabel("Timesteps")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()