# coding=utf-8

import ml.feature_extraction.eeg.windowed_extractor
import ml.feature_extraction.eeg.raw_extractor
import ml.feature_extraction.eeg.freq_extractor
import ml.feature_extraction.eeg.trash_extractor

import ml.feature_extraction.audio.acoustics_extractor

from .. import utils


def create(config):
    params = utils.read_config(config)
    if params["extraction_method"] == "raw":
        return ml.feature_extraction.eeg.raw_extractor.RawExtractor(params)
    elif params["extraction_method"] == "freqs":
        return ml.feature_extraction.eeg.freq_extractor.FreqExtractor(params)
    elif params["extraction_method"] == "trash":
        return ml.feature_extraction.eeg.trash_extractor.TrashExtractor(params)
    elif params["extraction_method"] == "window_mean":
        return ml.feature_extraction.eeg.windowed_extractor.WindowedExtractor(params)
    elif params["extraction_method"] == "wavelets":
        return ml.feature_extraction.eeg.wavelets_extractor.WaveletsExtractor(params)
    elif params["extraction_method"] == "opensmile":
        return ml.feature_extraction.audio.acoustics_extractor.AcousticsExtractor(params)
