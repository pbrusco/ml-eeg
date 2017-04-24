# coding: utf-8

import matplotlib.pyplot as plt
import math

from . import signal_processing

import matplotlib.patches as patches
import numpy as np
from sklearn import metrics
import seaborn as sns
import mne
from IPython.display import display


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


def plot_epochs_average(data, y_lim, tmin, window, freq, marks, epochs_max_number, ax):
    t0, tf = window
    t0_frame = signal_processing.frame_at(t0, freq, tmin)
    tf_frame = signal_processing.frame_at(tf, freq, tmin)

    epochs_mean = data[:, t0_frame:tf_frame, 0:epochs_max_number].mean(2)

    samples = epochs_mean.shape[1]

    for channel in range(0, epochs_mean.shape[0]):
        c_plt, = ax.plot([t0 + (s / freq) for s in range(0, samples)], epochs_mean[channel], "k", alpha=0.3, label="each channel")

    c2_plt, = ax.plot([t0 + (s / freq) for s in range(0, samples)], epochs_mean.mean(0), "r", linewidth=2.0, label="channels avg")
    set_plot(plt, y_lim, window, t0, tf, marks, ax)
    rect = draw_rectangle(plt, window, ax)

    ax.legend(handles=[c_plt, c2_plt, rect])


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
    ax.axvline(marks[0])
    ax.axvline(marks[1])


def draw_rectangle(plt, window, ax=None):
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
                             fill=None, label="stimulus")
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
    plt.show()
    return gr.get_figure()


def feature_importances_topomap(features_table, features_config, cmap="Greys", fontsize=15, title=""):
    montage = mne.channels.read_montage(features_config["montage"])

    vmin = features_table.feature_importances_folds_mean.min()
    vmax = features_table.feature_importances_folds_mean.max()
    l = mne.channels.make_eeg_layout(mne.create_info(montage.ch_names, features_config["freq"], ch_types="eeg", montage=montage))

    starting_samples = sorted(set(features_table.starting_sample))
    fig, axes = plt.subplots(1, len(starting_samples), figsize=(3 * len(starting_samples), 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    [ax.axis('off') for ax in axes]
    for sample, ax in zip(starting_samples, axes):
        sample_data = features_table[features_table["starting_sample"] == sample]

        t = list(sample_data.starting_time)[0]

        values = np.array(sample_data.groupby("channel")["feature_importances_folds_mean"].mean())
        mne.viz.plot_topomap(values, l.pos[:, 0:2], outlines="skirt", vmin=vmin, vmax=vmax, axes=ax, show_names=False, names=montage.ch_names, show=False, cmap=cmap)
        ax.set_title("{} ms".format(t), fontsize=fontsize)

    fig.suptitle(title, fontsize=16)

    plt.show()


def feature_importances_bars(features_table, title=""):
    plt.figure()
    # gr = sns.pointplot(x="starting_time", y="feature_importances_folds_mean",
    #                    data=features_table, hue="window_size", palette="Set1", markers=["D", "s", "o", ">"],
    #                    ci=68, scale=1, linestyles=["--"] * 4)
    #
    # gr.legend(title="window size", loc='upper center')
    # ticks = gr.get_xticklabels()
    #
    # gr.set_xticklabels(ticks, rotation=90)
    # gr.set_ylabel("mean(feature importance)")
    # # gr.set_yticks([])
    # gr.set_xlabel("window starting time (ms)")
    # gr.set_title(title)
    #
    # f = gr.get_figure()
    # f.tight_layout()
    for idx, row in features_table.iterrows():
        alpha = row.window_size / features_table.window_size.max()
        plt.hlines(y=row.feature_importances_folds_mean, lw=5, alpha=alpha, xmin=row.starting_time, xmax=row.end_time)

    plt.ylim([features_table.feature_importances_folds_mean.min(), features_table.feature_importances_folds_mean.max()])
    plt.xlim([features_table.starting_time.min(), features_table.end_time.max()])
    plt.show()
