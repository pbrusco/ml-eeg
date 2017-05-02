# coding: utf-8

import mne
import numpy as np


def from_eeglab_epochs_setfile(set_filename):
    data_mne = mne.io.read_epochs_eeglab(set_filename, events=None, event_id=None)
    return data_mne.info, data_mne.get_data().transpose((1, 2, 0)) * 1e6


def read_montage(montage_filename):
    montage = mne.channels.read_montage(montage_filename)
    new_locs = np.array([(-y, x, z) for (x, y, z) in montage.pos])
    montage.pos = new_locs
    return montage
