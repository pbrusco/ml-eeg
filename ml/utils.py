#!/usr/bin/python
# coding: utf-8
import numpy as np
import mne
import sys
import ConfigParser
import collections
from os.path import expanduser
import system
import csv
import os

HOME = expanduser("~")

mne.set_log_level('ERROR')


def read_config(filename):
    if not system.exists(filename):
        raise Exception("missing config: {}".format(filename))
    config = ConfigParser.SafeConfigParser()
    config.read([filename])
    res = collections.defaultdict(str)
    for k, v in config.items("DEFAULT"):
        res[k] = eval(v)
    return res


def read_list(filename):
    if not system.exists(filename):
        raise Exception("missing list: {}".format(filename))
    abspath = os.path.abspath(filename)
    dirpath = os.path.dirname(abspath)

    file = open(abspath, "r")

    lines = file.readlines()
    for line in lines:
        if line.strip() != "":
            l = line.replace("~/", HOME + "/")
            l = l.replace("./", dirpath + "/")
            yield l.split()


def read_wavesurfer(filename):
    return [(float(t0), float(tf), lbl) for (t0, tf, lbl) in read_list(filename)]


def read_ipus(vad_filename):
    return [(t0, tf) for (t0, tf, lbl) in read_wavesurfer(vad_filename) if lbl == "1"]


def read_turns(turns_filename):
    return [(t0, tf, tt) for (t0, tf, tt) in read_wavesurfer(turns_filename) if tt != "#"]


def read_csv(filename, delimiter=","):
    res = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter)
        for i, row in enumerate(spamreader):
            if i == 0:
                header = row
            else:
                res.append(dict(zip(header, row)))
    return res


def read_dat_table(filename, header_filename):
    with open(header_filename, "r") as header_file:
        content = header_file.read()
        header = [h.split()[0] for h in content.strip().split("\n")]

    data = []
    with open(filename, "r") as data_file:
        for line in data_file:
            row = dict(zip(header, line.strip().split("\t")))
            data.append(row)
    return data


def save_list(lines, filename):
    with open(filename, "w") as fn:
        for line in lines:
            if type(line) is list or type(line) is tuple:
                fn.write("\t".join(line) + "\n")
            else:
                fn.write(line + "\n")


def print_inline(str):
    delete = "\b" * (len(str) + 2)
    print "{0}{1}".format(delete, str),
    sys.stdout.flush()


def call_with_necesary_params_only(fn, params):
    arg_count = fn.func_code.co_argcount
    args = fn.func_code.co_varnames[:arg_count]

    args_dict = {}
    for k, v in params.iteritems():
        if k in args:
            args_dict[k] = v

    return fn(**args_dict)


def extend_dict(dictionary, key, new_data):
    if key not in dictionary:
        dictionary[key] = new_data
    else:
        dictionary[key] = np.concatenate((dictionary[key], new_data), axis=2)
    return dictionary


def get_param_or(args, arg_number, default):
    if len(args) > arg_number:
        return args[arg_number]
    else:
        return default


def count_if(f, values):
    return len([0 for v in values if f(v)])


def mappings():
    res = {}
    res["List_Speak"] = dict(listening_H=1, listening_S=1, listening_BC=1,
                             speaking_H=0, speaking_S=0, speaking_BC=0)

    res["H_S_listening"] = dict(listening_H=1, listening_S=0, listening_BC=None,
                                speaking_H=None, speaking_S=None, speaking_BC=None)

    res["H_S_speaking"] = dict(listening_H=None, listening_S=None, listening_BC=None,
                               speaking_H=1, speaking_S=0, speaking_BC=None)

    res["List_Speak_switch"] = dict(listening_H=None, listening_S=1, listening_BC=None,
                                    speaking_H=None, speaking_S=0, speaking_BC=None)

    res["List_Speak_hold"] = dict(listening_H=1, listening_S=None, listening_BC=None,
                                  speaking_H=0, speaking_S=None, speaking_BC=None)

    res["List_Speak_no_bc"] = dict(listening_H=1, listening_S=1, listening_BC=None,
                                   speaking_H=0, speaking_S=0, speaking_BC=None)
    return res


def apply_mapping_to_data(X, y, categories_mapping):
    mapped_y = np.array([categories_mapping[y_i] for y_i in y])
    X_filtered = X[mapped_y != np.array(None), :]
    y_filtered = mapped_y[mapped_y != np.array(None)].astype(int)

    return X_filtered, y_filtered


def compare_intersection(y_1, y_2):
    intersection = sum([1 if y else 0 for y in (y_1 == y_2)])
    print "permutation intersection: {}%".format(round((100.0 * intersection) / len(y_2), 2))
    return intersection


def subsample_ids(y):
    counts = collections.Counter(y)
    ids = []
    min_count = min(counts.values())

    counts = dict([(label, 0) for label in counts.keys()])
    for idx, label in enumerate(y):
        if counts[label] >= min_count:
            continue
        else:
            ids.append(idx)
            counts[label] += 1
    return ids


def unzip(arr):
    lst1, lst2 = zip(*arr)
    return list(lst1), list(lst2)
