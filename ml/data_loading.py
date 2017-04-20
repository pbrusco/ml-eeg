# coding: utf-8

from . import utils
from . import data_import
import numpy as np


# Specific data-loading for this project
def set_file_for(session_id, speaker_id, condition, set_files_folder):
    return set_files_folder + "/set-trimmed-trials-epoched-merged-FiltDur-{}/s{}-{}-Deci-Filter-Trim-ICA-Pruned-Epoched-Merged.set".format(condition, session_id, speaker_id)


def data_for_subject(session_id, speaker_id, conditions, corpora_folder, dummy_test=False, verbose=True):
    # En la siguiente sección se obtienen los datos directo de los archivos .set y .fdt y se separan en datos de
    # desarrollo y datos de test. Para esto, por sujeto, por condición, se seleccionan 4/5 de los epochs (de manera aleatoria)
    # para ser utilizados en desarrollo y el quinto restante para resultados finales.

    development_data = {}
    development_epoch_ids = []
    control_data = {}

    for condition in conditions:
        data_for_condition_for_subject = from_eeglab(session_id, speaker_id, condition, corpora_folder, verbose=verbose)  # channels x samples x epochs
        if dummy_test:
            print("generating random data")
            data_for_condition_for_subject = np.random.random(data_for_condition_for_subject.shape) + (conditions.index(condition) * 20)

        n_epochs = data_for_condition_for_subject.shape[2]
        # development_ids, control_ids = split_dev_test_sets(np.arange(n_epochs))
        development_ids = np.arange(n_epochs)
        development_epoch_ids.extend([(condition, i) for i in development_ids])

        development_data_for_condition = data_for_condition_for_subject[:, :, development_ids]
        # control_data_for_condition = data_for_condition_for_subject[:, :, control_ids]

        utils.extend_dict(development_data, condition, development_data_for_condition)
        # extend_dict(control_data, condition, control_data_for_condition)

    return (development_data, control_data, development_epoch_ids)


def from_eeglab(session_id, speaker_id, condition, set_files_folder, verbose=True):
    set_filename = set_file_for(session_id, speaker_id, condition, set_files_folder)
    info, data = data_import.from_eeglab_epochs_setfile(set_filename)
    if verbose:
        print("{}-{}: {} ({} x {} x {}) (channels x samples x epochs)".format(session_id, speaker_id, condition, info["nchan"], data.shape[1], data.shape[2]))

    return data


def split_dev_test_sets(epoch_ids):
    epoch_ids_partition = np.array_split(epoch_ids, 5)

    development_ids = np.concatenate(epoch_ids_partition[0:4])
    validation_ids = epoch_ids_partition[4]
    return development_ids, validation_ids
