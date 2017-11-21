# coding=utf-8



from .. import utils


def create(config):
    params = utils.read_config(config)
    if params["extraction_method"] == "raw":
        import ml.feature_extraction.eeg.raw_extractor
        return ml.feature_extraction.eeg.raw_extractor.RawExtractor(params)
    elif params["extraction_method"] == "freqs":
        import ml.feature_extraction.eeg.freq_extractor
        return ml.feature_extraction.eeg.freq_extractor.FreqExtractor(params)
    elif params["extraction_method"] == "trash":
        import ml.feature_extraction.eeg.trash_extractor
        return ml.feature_extraction.eeg.trash_extractor.TrashExtractor(params)
    elif params["extraction_method"] == "window_mean":
        import ml.feature_extraction.eeg.windowed_extractor
        return ml.feature_extraction.eeg.windowed_extractor.WindowedExtractor(params)
    elif params["extraction_method"] == "wavelets":
        return ml.feature_extraction.eeg.wavelets_extractor.WaveletsExtractor(params)
    elif params["extraction_method"] == "acustics":
        import ml.feature_extraction.audio.acoustics_extractor
        return ml.feature_extraction.audio.acoustics_extractor.AcousticsExtractor(params)
