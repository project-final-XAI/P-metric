import os
import random  # Import the entire random module
import numpy as np
import matplotlib.pyplot as plt


def show_heatmap():
    # Load the heatmap from a .npy file
    heatmaps_path = "results/heatmaps"

    # Choose 5 random heatmap files for demonstration
    heatmap_files = random.sample(os.listdir(heatmaps_path), 5)

    for file in heatmap_files:
        file_path = os.path.join(heatmaps_path, file)
        heatmap = np.load(file_path)

        # Display the heatmap
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Attribution Heatmap: {file}")
        plt.show()


if __name__ == "__main__":
    show_heatmap()