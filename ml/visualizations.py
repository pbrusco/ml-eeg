# coding: utf-8


from . import signal_processing
from . import data_import

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import seaborn as sns

from IPython.display import display
import math


from sklearn import metrics
import numpy as np
import mne


def plot_roc_curve(y_actual, y_scores, ax, is_permutation, fontsize=30):
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_scores)
    roc_auc = metrics.auc(fpr, tpr)

    if is_permutation:
        ax.plot(fpr, tpr, "b-", alpha=0.5, lw=0.3)
    else:
        ax.plot([0, 1], [0, 1], '--', lw=6, color="k", label='Chance')
        ax.plot(fpr, tpr, "g-", alpha=1, lw=8, label='ROC (area = {})'.format("{:.3f}".format(roc_auc)))
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        plt.tick_params(axis='both', which='major', labelsize=30)
        ax.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=fontsize)


def subject_table(data):
    table = data.reindex_axis(["session",
                               "experiment",
                               "auc",
                               "auc_p_val",
                               "acc",
                               "acc_p_val",
                               "support",
                               ], axis=1)
    if not table.empty:
        s = table.style\
                 .applymap(_set_color, subset=["auc_p_val", "acc_p_val"])\
                 .applymap(_bold, subset=["session"])
        display(s)


def _set_color(val):
    if math.isnan(val):
        color = "rgba(255, 255, 255, 1)"
    elif val < 0.05:
        color = "rgba(216, 246, 206, 0.5)"
    elif val <= 0.1:
        color = "rgba(242, 245, 169, 0.5)"
    else:
        color = "rgba(245, 169, 169, 0.3)"
    return 'background-color: %s;' % color


def _bold(val):
    return 'font-weight: bold'


def _bar(val):
    if val > 0.6:
        return _bar_color(val, "green")
    elif val > 0.4:
        return _bar_color(val, "yellow")
    else:
        return _bar_color(val, "red")


def _bar_2(val):
    return _bar_color(val, "blue")


def _bar_color(val, color):
        base = 'width: 10em; height: 80%;'
        attrs = (base + 'background: linear-gradient(90deg,rgba{c} {w}%, '
                        'transparent 0%)')

        color = sns.light_palette(color, as_cmap=True)(val)
        c = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 0.3)
        return attrs.format(c=c, w=val * 100)


def plot_epochs_average(data, y_lim, tmin, window, freq, marks=[], ax=None, color="red", label="epochs mean"):
    # Data shape: (samples, trial)
    if not ax:
        ax = plt.gca()

    t0, tf = window
    t0_frame = signal_processing.frame_at(t0, freq, tmin)
    tf_frame = signal_processing.frame_at(tf, freq, tmin)

    samples = data.shape[0]

    for epoch_id in range(0, data.shape[1]):
        c_plt, = ax.plot([t0 + (s / freq) for s in range(0, samples)], data[:, epoch_id], color=color, alpha=0.05)

    epochs_mean = data[t0_frame:tf_frame, :].mean(1)

    c2_plt, = ax.plot([t0 + (s / freq) for s in range(0, samples)], epochs_mean, color=color, linewidth=2.0, label=label)
    set_plot(plt, y_lim, window, t0, tf, marks, ax)
    draw_rectangle(plt, window, ax, label=None)


def plot_epochs_comparition(data_dict, y_lim, tmin, window, freq, marks, epochs_max_number):
    t0, tf = window
    t0_frame = signal_processing.frame_at(t0, freq, tmin)
    tf_frame = signal_processing.frame_at(tf, freq, tmin)
    plt.figure()

    for condition in list(data_dict.keys()):
        epochs_mean = data_dict[condition][:, t0_frame:tf_frame, 0:epochs_max_number].mean(2)
        samples = epochs_mean.shape[1]

        plt.plot([t0 + (s / freq) for s in range(0, samples)], epochs_mean.mean(0), linewidth=2.0, label=condition)

    set_plot(plt, y_lim, window, t0, tf, marks, plt)
    draw_rectangle(plt, window)
    plt.legend()


def set_plot(plt, y_lim, window, t0, tf, marks, ax):
    plt.ylim(y_lim)
    plt.xlim(window)
    # plt.xticks([x/1000.0 for x in range(-2000, 101, 100) if (x/1000.0)>=t0 and (x/1000.0)<=tf])
    ax.axvline(marks[0], color="black")
    ax.axvline(marks[1], color="black")


def draw_rectangle(plt, window, ax=None, label="stimulus"):
    if not ax:
        ax = plt.gca()
    rect = patches.Rectangle((window[0], -40), width=-window[0] - 0.4, height=80,
                             color='grey',
                             alpha=0.5)
    ax.add_patch(rect)

    rect = patches.Rectangle((0, -40), width=window[1], height=80,
                             color='grey',
                             alpha=0.5)
    ax.add_patch(rect)

    rect = patches.Rectangle((-0.4, -40), width=0.4, height=80,
                             fill=None, edgecolor="black", label=label)
    ax.add_patch(rect)
    return rect


def barplot(session, data, measure):
    plt.figure()

    sns.set(style="white", context="talk", font_scale=1.0)
    sns.despine(bottom=True)
    no_perumutations = data[not data.experiment.str.endswith("permutation")]
    session_data = no_perumutations[no_perumutations.session == session]

    x = list(session_data.experiment)
    y = list(session_data[measure])
    hue = list(session_data.extraction_method)
    permutations = list(session_data[measure + "_permutation"])
    supports = list(session_data.support)

    bar = sns.barplot(x=x,
                      y=y,
                      palette="Set1",
                      hue=hue,
                      data=session_data)

    rectangles = []
    for i, p in enumerate(bar.patches):
        height = p.get_height()
        xmin = p.get_x()
        width = p.get_width()
        bar.text(xmin, height + 0.01, '%1.2f' % (y[i]), fontsize=14)
        rectangles.append(
            patches.Rectangle(
                (xmin, permutations[i]),   # (x,y)
                width,          # width
                0.01,          # height
                color="black",
            )
        )
        bar.text(p.get_x() + 0.1, 0.1, "N={} ".format(supports[i]), rotation=90, fontsize=14, backgroundcolor="w")

    for rectangle in rectangles:
        bar.add_patch(rectangle)
    plt.ylabel("AUC")
    plt.ylim([0, 1])
    plt.title("Session {}".format(session))
    plt.tight_layout()


def feature_importances_by_window_size(features_table, title):
    # https://gist.github.com/wmvanvliet/6d7c78ea329d4e9e1217
    gr = sns.stripplot(x="starting_time", y="feature_importances_folds_mean", data=features_table, hue="window_size", palette="Set2")
    gr.set_xticklabels(gr.get_xticklabels(), rotation=90)
    gr.set_title(title)
    plt.draw()
    return gr.get_figure()


def topomap(values_by_time, montage_file, freq, cmap="Greys", fontsize=15, title=""):
    montage = data_import.read_montage(montage_file)

    vmin = values_by_time.feature_importances_folds_mean.min()
    vmax = values_by_time.feature_importances_folds_mean.max()
    # vmin, vmax = (0.0005, 0.0015)
    l = mne.channels.make_eeg_layout(mne.create_info(montage.ch_names, freq, ch_types="eeg", montage=montage))

    times = sorted(set(values_by_time.time))
    fig, axes = plt.subplots(1, len(times), figsize=(3 * len(times), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    [ax.axis('off') for ax in axes]
    for top_n, (time, ax) in enumerate(zip(times, axes)):
        time_data = values_by_time[values_by_time["time"] == time]

        t = list(time_data.time)[0]
        image, _ = mne.viz.plot_topomap(list(time_data["values"]), l.pos[:, 0:2], vmin=vmin, vmax=vmax, outlines="skirt", axes=ax, show_names=False, names=l.names, show=False, cmap=cmap)
        if top_n == len(axes) - 1:
            divider = make_axes_locatable(ax)
            ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(image, cax=ax_colorbar)

        ax.set_title("{} ms".format(t), fontsize=fontsize)

    fig.suptitle(title, fontsize=16)
    plt.draw()


def lines(features_table, title=""):
    plt.figure()

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    for idx, row in features_table.iterrows():
        alpha = row.window_size / features_table.window_size.max()
        plt.hlines(y=row.feature_importances_folds_mean, lw=5, alpha=alpha, xmin=row.starting_time, xmax=row.end_time)

    plt.ylim([features_table.feature_importances_folds_mean.min(), features_table.feature_importances_folds_mean.max()])
    plt.xlim([features_table.starting_time.min(), features_table.end_time.max()])
    plt.draw()


def window_bars(features_table, title="", fontsize=20):
    features_table.sort_values(["window_size", "starting_time"], ascending=False, inplace=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap = matplotlib.cm.get_cmap('Greys')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    vmin, vmax = (features_table.feature_importances_folds_mean.min(), features_table.feature_importances_folds_mean.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, (_, row) in enumerate(features_table.iterrows()):
        val = row.feature_importances_folds_mean
        # plt.hlines(y=idx, lw=3, color=cmap(norm(val)), xmin=row.starting_time, xmax=row.end_time)
        p = patches.Rectangle(
            (row.starting_time, idx),  # (x, y)
            row.window_size,  # width
            1,  # height
            facecolor=cmap(norm(val)),
            # edgecolor="blue"
        )
        ax.add_patch(p)

    ax.set_title(title, fontsize=fontsize)
    plt.xlim([features_table.starting_time.min(), features_table.end_time.max()])
    plt.ylim([-1, len(features_table) + 2])

    divider = make_axes_locatable(ax)
    ax_colorbar = divider.append_axes('right', size='60%', pad=0.01)

    img = plt.imshow(np.array([[vmin, vmax]]), cmap=cmap)
    img.set_visible(False)
    plt.colorbar(img, cax=ax_colorbar, orientation="vertical")

    plt.draw()
