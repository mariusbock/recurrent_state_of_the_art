import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


class HandlerEllipse(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_sensordata_and_labels(sensordata=None, columns=None, predictions_path='', datasetname=''):
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import StrMethodFormatter

    acc_x = []
    acc_y = []
    acc_z = []
    subjects = []
    labels = []
    if datasetname == "WETLAB":
        datasetname = "WETLAB WITH ALTERNATIVE LABELS"
        sensordata.columns = ['subject', 'acc_x', 'acc_y', 'acc_z', 'label', 'altlabel']
    else:
        sensordata.columns = ['subject', 'acc_x', 'acc_y', 'acc_z', 'label']
    if sensordata is None:
        print("input data is empty, exiting the program.")
        exit(0)
    if isinstance(sensordata, pd.DataFrame):
        acc_x = sensordata["acc_x"].to_numpy()
        acc_y = sensordata["acc_y"].to_numpy()
        acc_z = sensordata["acc_z"].to_numpy()
        subjects = sensordata["subject"].to_numpy()
        labels = sensordata["altlabel"].to_numpy()
    if sensordata is np.ndarray and columns is None:
        columns = ["subject", "acc_x", "acc_y", "acc_z", "label"]
        acc_x = sensordata[1]
        acc_y = sensordata[2]
        acc_z = sensordata[3]
        subjects = sensordata[0]
        labels = sensordata[4]
    if sensordata is np.ndarray and columns is not None:

        acc_x = list(sensordata[1])
        acc_y = list(sensordata[2])
        acc_z = list(sensordata[3])
        subjects = sensordata[0]
        labels = sensordata[4]

    n_classes = len(np.unique(labels))
    n_subjects = len(np.unique(subjects))
    # plot 1:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(acc_x, color='blue')
    ax1.plot(acc_y, color='green')
    ax1.plot(acc_z, color='red')
    ax1.set_ylabel("Acceleration (mg)")
    ax1.legend(["acc x", "acc y", "acc z"])

    unordered_unique_labels, first_occurences, labels_onehot = np.unique(labels, return_inverse=True,
                                                                         return_index=True)
    order = []

    ordered_unique_onehot_labels = first_occurences.copy()
    ordered_unique_onehot_labels.sort()
    ordered_labels = []

    for index in ordered_unique_onehot_labels:
        ordered_labels.append(unordered_unique_labels[np.where(first_occurences == index)[0][0]])
        order.append(np.where(first_occurences == index)[0][0])
    colors1 = sns.color_palette(palette="hls", n_colors=n_classes).as_hex()
    colors2 = sns.color_palette(palette="Spectral", n_colors=n_subjects).as_hex()
    original_colors = colors1.copy()

    for i in range(0, n_classes):
        colors1[i] = original_colors[order[i]]

    cmap1 = LinearSegmentedColormap.from_list(name='My Colors1', colors=colors1, N=len(colors1))
    cmap2 = LinearSegmentedColormap.from_list(name='My Colors2', colors=colors2, N=len(colors2))
    ax2.set_yticks([])

    if predictions_path != '':
        predictions = pd.read_csv(predictions_path)
        ax2.set_ylabel("Ground Truth vs Predictions")
        ax2.pcolor([labels_onehot, predictions], cmap=cmap1)
    else:
        ax2.set_ylabel("Ground Truth")
        ax2.pcolor([labels_onehot], cmap=cmap1)

    ax3.set_yticks([])
    ax3.set_ylabel("Subjets")
    ax3.pcolor([subjects], cmap=cmap2)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(datasetname)

    c = [mpatches.Circle((0.5, 0.5), radius=0.25, facecolor=colors1[i], edgecolor="none") for i in
         range(n_classes)]

    plt.legend(c, unordered_unique_labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=n_classes,
               fancybox=True, shadow=True,
               handler_map={mpatches.Circle: HandlerEllipse()}).get_frame()

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
    plt.show()


def get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt

    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

data = pd.read_csv('/Users/alexander/Downloads/wetlab_data.csv')
plot_sensordata_and_labels(data, [], datasetname='WETLAB')
