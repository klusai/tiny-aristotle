import matplotlib.pyplot as plt
import re
import numpy as np
import argparse


def parse_log_file(log_file):
    """
    Parses the log file to extract training and validation losses, perplexities, and timing.

    Parameters:
        log_file (str): Path to the log file.

    Returns:
        dict: Parsed data containing iteration numbers, train_loss, val_loss, train_ppl, val_ppl.
    """
    data = {
        "iteration": [],
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
    }

    train_pattern = re.compile(
        r"iter=(\d+)/\d+ train_loss=([0-9\.]+) train_ppl=([0-9\.]+)"
    )
    val_pattern = re.compile(r"iter=(\d+)/\d+ val_loss=([0-9\.]+) val_ppl=([0-9\.]+)")

    with open(log_file, "r") as f:
        for line in f:
            train_match = train_pattern.search(line)
            if train_match:
                data["iteration"].append(int(train_match.group(1)))
                data["train_loss"].append(float(train_match.group(2)))
                data["train_ppl"].append(float(train_match.group(3)))

            val_match = val_pattern.search(line)
            if val_match:
                # Append None for iterations without validation data
                if len(data["val_loss"]) < len(data["iteration"]):
                    data["val_loss"].append(None)
                    data["val_ppl"].append(None)

                data["val_loss"][-1] = float(val_match.group(2))
                data["val_ppl"][-1] = float(val_match.group(3))

    # Fill remaining val_loss and val_ppl with None if not provided
    while len(data["val_loss"]) < len(data["iteration"]):
        data["val_loss"].append(None)
        data["val_ppl"].append(None)

    return data


def plot_training_progress(data, save_path=None):
    """
    Plots training loss, validation loss, and perplexity over iterations.

    Parameters:
        data (dict): Parsed data containing iteration numbers, train_loss, val_loss, train_ppl, val_ppl.
        save_path (str): Path to save the plot (optional).
    """
    iterations = data["iteration"]

    # Training and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(
        iterations, data["train_loss"], label="Train Loss", color="blue", linewidth=1.5
    )
    if any(v is not None for v in data["val_loss"]):
        plt.plot(
            iterations,
            [v if v is not None else np.nan for v in data["val_loss"]],
            label="Validation Loss",
            color="orange",
            linestyle="--",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_loss.pdf", format="pdf")
    plt.show()

    # Training and Validation Perplexity
    plt.figure(figsize=(8, 6))
    plt.plot(
        iterations,
        data["train_ppl"],
        label="Train Perplexity",
        color="blue",
        linewidth=1.5,
    )
    if any(v is not None for v in data["val_ppl"]):
        plt.plot(
            iterations,
            [v if v is not None else np.nan for v in data["val_ppl"]],
            label="Validation Perplexity",
            color="orange",
            linestyle="--",
            linewidth=1.5,
            marker="o",
            markersize=4,
        )
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.title("Training and Validation Perplexity", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6, color="gray")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_perplexity.pdf", format="pdf")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training progress plots from log data."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to the log file containing training and validation metrics.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the generated plots. If not provided, plots will not be saved.",
    )

    args = parser.parse_args()

    data = parse_log_file(args.log_file)
    plot_training_progress(data, save_path=args.save_path)


if __name__ == "__main__":
    main()
