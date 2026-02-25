#!/usr/bin/env python3
"""
Visualize MNIST digit samples
Requires: numpy, matplotlib, gzip
"""

import gzip
import numpy as np
import matplotlib.pyplot as plt
import os


def load_mnist_images(filename):
    """Load MNIST images from gz file"""
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(filename):
    """Load MNIST labels from gz file"""
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def visualize_samples(images, labels, num_samples=20):
    """Visualize MNIST samples in a grid"""
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    fig.suptitle("MNIST Dataset Samples", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i], cmap="gray")
            ax.set_title(f"Label: {labels[i]}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("images/mnist_samples.png", dpi=150, bbox_inches="tight")
    print("Saved: images/mnist_samples.png")


def main():
    print("Loading MNIST dataset...")
    images = load_mnist_images("data/train-images-idx3-ubyte.gz")
    labels = load_mnist_labels("data/train-labels-idx1-ubyte.gz")

    print(f"Loaded {len(images)} images")
    print("Creating visualization...")

    visualize_samples(images, labels)
    print("Done!")


if __name__ == "__main__":
    main()
