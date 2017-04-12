#!/usr/bin/python
# coding: utf-8


class FeatureExtractor(object):
    def extract(self, instance):
        raise NotImplementedError("{} must implement the 'extract' method".format(self))

    def __init__(self, params):
        raise NotImplementedError("{} must implement the '__init__' method".format(self))

    def batch_extract(self, instances_list):
        raise NotImplementedError("{} must implement the 'extract_batch' method".format(self))

import eeg_feature_extraction
import audio_feature_extraction


def create(**params):
    if params["extraction_method"] == "raw":
        return eeg_feature_extraction.RawExtractor(params)
    elif params["extraction_method"] == "freqs":
        return eeg_feature_extraction.FreqExtractor(params)
    elif params["extraction_method"] == "trash":
        return eeg_feature_extraction.TrashExtractor(params)
    elif params["extraction_method"] == "window_mean":
        return eeg_feature_extraction.WindowedExtractor(params)
    elif params["extraction_method"] == "wavelets":
        return eeg_feature_extraction.WaveletsExtractor(params)
    elif params["extraction_method"] == "opensmile":
        return audio_feature_extraction.AcousticsExtractor(params)
