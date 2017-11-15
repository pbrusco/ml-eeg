# coding=utf-8


from .. import feature_extraction

import ml.utils as utils
import numpy as np


class EEGFeatureExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config, require=["trial_tmin", "trial_tmax", "sfreq", "channels_to_extract_from", "extraction_tmin", "extraction_tmax"])

        self.trial_tmin = float(self.params["trial_tmin"])
        self.trial_tmax = float(self.params["trial_tmax"])
        self.sfreq = float(self.params["sfreq"])
        self.n_samples = int((self.trial_tmax - self.trial_tmin) * self.sfreq)
        self.n_channels = self.params["n_channels"]

        self.extraction_tmin = self.params["extraction_tmin"]
        self.extraction_tmax = self.params["extraction_tmax"]
        self.extraction_time_limits = [self.extraction_tmin, self.extraction_tmax]

        self.channels_to_extract_from = np.array(self.params["channels_to_extract_from"])
        self.verbose = self.params["verbose"] if ("verbose" in self.params) else None

        self.validate_extractor()

        if self.verbose:
            print("extractor params")
            print((self.__dict__))

    def validate_extractor(self):
        msg = "The numbers of channel to extract from are not in range"
        assert np.all((self.channels_to_extract_from >= 0) & (self.channels_to_extract_from < self.n_channels)), msg

        msg = "times are not in order"
        assert self.trial_tmin < self.trial_tmax, msg

        msg = "extraction time is not valid (should be contained in [tmin, tmax] range)"
        in_range = self.extraction_tmin >= self.trial_tmin and \
                   self.extraction_tmin < self.trial_tmax and \
                   self.extraction_tmax > self.trial_tmin and \
                   self.extraction_tmax <= self.trial_tmax and \
                   self.extraction_tmin < self.extraction_tmax

        assert in_range, msg
