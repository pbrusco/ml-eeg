from .eeg_feature_extractor import EEGFeatureExtractor


class WaveletsExtractor(EEGFeatureExtractor):
    def __init__(self, config):
        super(WaveletsExtractor, self).__init__(config)

    def extract(self, trial):
        pass
