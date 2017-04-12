#!/usr/bin/python
# coding: utf-8

import mne


def from_eeglab_epochs_setfile(set_filename, order="F"):
    data_mne = mne.io.read_epochs_eeglab(set_filename, events=None, event_id=None)
    return data_mne.info, data_mne.get_data().transpose((1, 2, 0)) * 1e6
