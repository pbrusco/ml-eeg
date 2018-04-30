# coding: utf-8

import os
import os.path
import subprocess
from time import gmtime, strftime

LOGLEVEL = 3

def mute():
     set_log_level(1)

def set_log_level(lvl):
    # 1 errores
    # 2 errores + warnings
    # 3 errores + warnings + info
    global LOGLEVEL
    LOGLEVEL=lvl

def home():
    return os.path.expanduser("~")


def now():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def exists(fname):
    return os.path.isfile(fname)


def basename(dir):
    if dir.endswith("/"):
        dir = dir[:-1]
    return os.path.basename(dir)


def cp(src, dst):
    run_external_command("cp", non_named_params=[src, dst])


def rm(filename):
    run_external_command("rm", non_named_params=[filename])


def run_command(cmd, skip=False, verbose=True):
    if verbose:
        if not skip:
            info("(running) {}".format(cmd))
        else:
            warning("(skiping) {}".format(cmd))
    if not skip:
        try:
            return subprocess.check_output(cmd, shell=True).decode("utf-8")
        except subprocess.CalledProcessError as e:
            error(cmd)
            raise e


def run_external_command(cmd, **args):
    if "skip" in args:
        skip = args["skip"]
        del args["skip"]
    else:
        skip = False

    if "non_named_params" in args:
        no_named = " " + " ".join([str(p) for p in args["non_named_params"]])
        del args["non_named_params"]
    else:
        no_named = ""

    for (arg, val) in list(args.items()):
        cmd += " --{} {}".format(arg, val)

    cmd += no_named
    return run_command(cmd, skip=skip)


def run_script(module, **args):
    # Run a script file without loading the environment again (should be imported and have a main function)
    cmd = "./scripts/"
    cmd += module.__name__.replace("scripts.", "")
    cmd += ".py"

    if "skip" in args:
        skip = args["skip"]
        del args["skip"]
    else:
        skip = False

    for (arg, val) in list(args.items()):
        cmd += " --{} '{}'".format(arg, val)

    if skip:
        warning("(skiping) {}".format(cmd))
    else:
        info("(running) {}".format(cmd))
        try:
            return module.main(**args)
        except Exception:
            error(("Running {} with args {}".format(module.__name__, args)))
            raise


def warning(message, background=True):
    global LOGLEVEL
    if LOGLEVEL > 1:
        if background:
            print(('\n (WARNING) \x1b[2;33;40m' + message + '\x1b[0m'))
        else:
            print(('\n (WARNING) ' + message))

def error(message, background=True):
    global LOGLEVEL
    if LOGLEVEL > 0:
        if background:
            print(('\n (ERROR) \x1b[2;31;43m' + message + '\x1b[0m'))
        else:
            print(('\n (ERROR) ' + message))

def info(message, background=True):
    global LOGLEVEL
    if LOGLEVEL > 2:
        if background:
            print(('\n (INFO) \x1b[2;32;40m' + message + '\x1b[0m'))
        else:
            print(('\n (INFO) ' + message))
