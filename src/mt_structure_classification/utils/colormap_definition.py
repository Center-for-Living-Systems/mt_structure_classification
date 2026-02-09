import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap


def make_repeated_tab10_cmap(n_labels: int = 200) -> ListedColormap:
    """
    Builds a colormap with many distinct-ish labels by repeating tab10 colors.
    Index 0 is black/transparent-ish.
    """
    base = cm.get_cmap("tab10", 10).colors  # (10,4)
    colors = base.copy()
    while colors.shape[0] < (n_labels + 1):
        colors = np.concatenate([colors, base], axis=0)

    colors = colors[:n_labels]  # for labels 1..n_labels
    colors = np.concatenate([np.zeros((1, 4)), colors], axis=0)  # label 0
    colors[0, 3] = 1.0
    return ListedColormap(colors)
