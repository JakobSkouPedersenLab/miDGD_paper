from sklearn.model_selection import StratifiedKFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_kfold(X, n_splits=5, figsize=(12, 6)):

    skf = StratifiedKFold(n_splits=n_splits) 

    fig, ax = plt.subplots(figsize=figsize)  # Increased figure size for better clarity

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(skf.split(X, X['cancer_type'])):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results using seaborn scatterplot
        sns.scatterplot(
            x=np.arange(len(indices)),
            y=np.full(len(indices), ii + 0.5),
            hue=indices,
            palette=['gray', 'orange'],  # Simplified color palette
            linewidth=0,  # Removed line width to make individual points stand out
            s=10,  # Adjusted size of points
            ax=ax,
            legend=False  # Removed legend for simplicity
        )

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
    )
    ax.grid(True)  # Added grid for better readability
    plt.tight_layout()  # Adjust layout to fit everything neatly
    plt.show()