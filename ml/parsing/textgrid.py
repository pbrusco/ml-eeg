# coding=utf-8

import ml.system as system
import textgrid


def read(textgrid_file):
    if not system.exists(textgrid_file):
        raise Exception("Missing file: " + textgrid_file)

    return textgrid.TextGrid().read(textgrid_file)
