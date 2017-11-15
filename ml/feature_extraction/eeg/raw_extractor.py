from .eeg_feature_extractor import EEGFeatureExtractor


class RawExtractor(EEGFeatureExtractor):
    def __init__(self, config):
        super(RawExtractor, self).__init__(config)

    def extract(self, trial):
        assert self.n_channels == trial.shape[0], "the number of channels in the extractor configuration doen't match {} vs {}".format(self.n_channels, trial.shape[0])
        ti, tf = self.extraction_time_limits

        initial_frame = int((ti - self.trial_tmin) * self.sfreq)
        end_frame = int((tf - self.trial_tmin) * self.sfreq)
        data = trial[self.channels_to_extract_from, initial_frame:end_frame]
        return data

    def feature_names(self):
        pass
