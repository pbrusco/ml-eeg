# coding: utf-8

import mne
import numpy as np
from . import system


def from_eeglab_raw_setfile(set_filename):
    data_mne = mne.io.read_raw_eeglab(set_filename)
    data_mne.load_data()
    d = data_mne._data * 1e6
    info = data_mne.info

    system.info("Loading data from {} (channels={}, samples={})".format(set_filename, d.shape[0], d.shape[1]))
    system.info(str(info))
    return info, d


def from_eeglab_epochs_setfile(set_filename):
    data_mne = mne.io.read_epochs_eeglab(set_filename, events=None, event_id=None)
    d = data_mne.get_data() * 1e6
    info = data_mne.info

    system.info("Loading data from {} (epochs={}, channels={}, samples={})".format(set_filename, d.shape[0], d.shape[1], d.shape[2]))
    system.info(str(info))

    return info, d


def read_montage(montage_filename):
    montage = mne.channels.read_montage(montage_filename)
    new_locs = np.array([(-y, x, z) for (x, y, z) in montage.pos])
    montage.pos = new_locs
    return montage
