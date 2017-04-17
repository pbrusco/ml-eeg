from time import gmtime, strftime
import os
import errno
import subprocess


def flatten(list_of_lists):
    return [e for l in list_of_lists for e in l]


def mkdir_p(folder):
    try:
        os.makedirs(folder)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(folder):
            pass


def now():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def exists(fname):
    return os.path.isfile(fname)


def sys_call(command, testing=False):
    print(command)
    if not testing:
        return subprocess.check_output(command, shell=True)


def show(label, data):
    print((label + ": " + str(data)))
