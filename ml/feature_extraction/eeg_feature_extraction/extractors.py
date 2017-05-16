# coding=utf-8

from . import eeg_feature_extraction
from . import audio_feature_extraction
from . import utils


def create(config):
    params = utils.read_config(config)
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
