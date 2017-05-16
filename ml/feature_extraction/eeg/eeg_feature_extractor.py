# coding=utf-8
from __future__ import division

from .. import feature_extraction

import ml.utils as utils
import numpy as np


class EEGFeatureExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config)

        self.trial_tmin = self.params["trial_tmin"]
        self.trial_tmax = self.params["trial_tmax"]
        self.sfreq = self.params["sfreq"]
        self.n_samples = int((self.trial_tmax - self.trial_tmin) * self.sfreq)
        self.n_channels = self.params["n_channels"]

        self.extraction_time_limits = self.params["extraction_time_limits"]
        self.channels_to_extract_from = np.array(self.params["channels_to_extract_from"])
        self.verbose = self.params["verbose"] if ("verbose" in self.params) else None

        self.validate_extractor()

        if self.verbose:
            print "extractor params"
            print(self.__dict__)

    def validate_extractor(self):
        msg = "The numbers of channel to extract from are not in range"
        assert np.all((self.channels_to_extract_from >= 0) & (self.channels_to_extract_from < self.n_channels)), msg

        msg = "times are not in order"
        assert self.trial_tmin < self.trial_tmax, msg

        msg = "extraction time is not valid (should be contained in [tmin, tmax] range)"
        in_range = self.extraction_time_limits[0] >= self.trial_tmin and \
                   self.extraction_time_limits[0] < self.trial_tmax and \
                   self.extraction_time_limits[1] > self.trial_tmin and \
                   self.extraction_time_limits[1] <= self.trial_tmax and \
                   self.extraction_time_limits[0] < self.extraction_time_limits[1]

        assert in_range, msg
