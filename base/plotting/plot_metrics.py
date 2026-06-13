import matplotlib.pyplot as plt
import seaborn as sns

def plot_dgd_metrics(loss_tab, figsize=(20, 12), main_title=""):
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.subplot(1, 2, 1)
    sns.lineplot(x="epoch", y="train_recon", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_recon", data=loss_tab, label="Test")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")

    plt.subplot(1, 2, 2)
    sns.lineplot(x="epoch", y="train_gmm", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_gmm", data=loss_tab, label="Test")
    plt.title("GMM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("GMM Loss")

    # Add the main title to the figure
    fig.suptitle(main_title, fontsize=24)
    
    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_metrics(loss_tab, figsize=(20, 12), main_title=""):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create a figure to hold the subplots
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Increase hspace to 0.5

    plt.subplot(2, 3, 1)
    sns.lineplot(x="epoch", y="train_mse", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_mse", data=loss_tab, label="Test")
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")

    plt.subplot(2, 3, 2)
    sns.lineplot(x="epoch", y="train_mae", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_mae", data=loss_tab, label="Test")
    plt.title("MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    plt.subplot(2, 3, 3)
    sns.lineplot(x="epoch", y="train_r2", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_r2", data=loss_tab, label="Test")
    plt.title("R^2")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")

    plt.subplot(2, 3, 4)
    sns.lineplot(x="epoch", y="train_spearman", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_spearman", data=loss_tab, label="Test")
    plt.title("Spearman Corr")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")

    plt.subplot(2, 3, 5)
    sns.lineplot(x="epoch", y="train_pearson", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_pearson", data=loss_tab, label="Test")
    plt.title("Pearson Corr")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson Corr")


    plt.subplot(2, 3, 6)
    sns.lineplot(x="epoch", y="train_expl_var", data=loss_tab, label="Train")
    sns.lineplot(x="epoch", y="test_expl_var", data=loss_tab, label="Test")
    plt.title("Expl Var")
    plt.xlabel("Epoch")
    plt.ylabel("Expl Var")

    # Add the main title to the figure
    fig.suptitle(main_title, fontsize=24)

    # Adjust the spacing between the main title and subplots
    plt.subplots_adjust(top=0.9)

    # Display the plots
    plt.show()