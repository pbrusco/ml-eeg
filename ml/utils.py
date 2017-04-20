# coding: utf-8

from __future__ import print_function

import numpy as np
import sys
import configparser
import collections
from os.path import expanduser
from . import system
import csv

HOME = expanduser("~")


def read_config(config_input):
    if isinstance(config_input, basestring):
        if not system.exists(config_input):
            raise Exception("missing config: {}".format(config_input))
        config = configparser.SafeConfigParser()
        config.read([config_input])
        res = collections.defaultdict(str)
        for k, v in config.items("DEFAULT"):
            res[k] = eval(v)
        return res
    elif isinstance(config_input, dict):
        return config_input
    else:
        raise Exception("config type not allowed (type={})".format(type(config_input)))


def read_list(filename):
    if not system.exists(filename):
        raise Exception("missing list: {}".format(filename))

    file = open(filename, "r")

    res = []
    lines = file.readlines()
    for line in lines:
        if line.strip() != "":
            l = line.replace("~/", HOME + "/")
            chunks = l.split()
            if len(chunks) > 1:
                res.append(chunks)
            else:
                res.append(chunks[0])

    return list(res)


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
                res.append(dict(list(zip(header, row))))
    return res


def read_dat_table(filename, header_filename):
    with open(header_filename, "r") as header_file:
        content = header_file.read()
        header = [h.split()[0] for h in content.strip().split("\n")]

    data = []
    with open(filename, "r") as data_file:
        for line in data_file:
            row = dict(list(zip(header, line.strip().split("\t"))))
            data.append(row)
    return data


def absolute_to_user_path(filename):
    return filename.replace(HOME + "/", "~/")


def save_list(lines, filename, verbose=True):
    with open(filename, "w") as fn:
        for line in lines:
            if type(line) is list or type(line) is tuple:
                fn.write("\t".join(line) + "\n")
            else:
                fn.write(line + "\n")
    if verbose:
        system.info("output list saved at {}".format(filename))


def print_inline(str):
    delete = "\b" * (len(str) + 2)
    txt = "{0}{1}".format(delete, str)

    print(txt, end=" ")

    sys.stdout.flush()


def call_with_necesary_params_only(fn, params):
    arg_count = fn.__code__.co_argcount
    args = fn.__code__.co_varnames[:arg_count]

    args_dict = {}
    for k, v in params.items():
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


def apply_mapping_to_data(X, y, categories_mapping):
    mapped_y = np.array([categories_mapping[y_i] for y_i in y])
    X_filtered = X[mapped_y != np.array(None), :]
    y_filtered = mapped_y[mapped_y != np.array(None)].astype(int)

    return X_filtered, y_filtered


def compare_intersection(y_1, y_2):
    intersection = sum([1 if y else 0 for y in (y_1 == y_2)])
    print("permutation intersection: {}%".format(round((100.0 * intersection) / len(y_2), 2)))
    return intersection


def subsample_ids(y):
    counts = collections.Counter(y)
    ids = []
    min_count = min(counts.values())

    counts = dict([(label, 0) for label in list(counts.keys())])
    for idx, label in enumerate(y):
        if counts[label] >= min_count:
            continue
        else:
            ids.append(idx)
            counts[label] += 1
    return ids


def unzip(arr):
    lst1, lst2 = list(zip(*arr))
    return list(lst1), list(lst2)


def flatten(list_of_lists):
    return [e for l in list_of_lists for e in l]
