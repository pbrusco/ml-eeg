# coding: utf-8
from .. import feature_extraction
import ml.feature_extraction.audio.tt_features_extractor

import ml.parsing.arff
import ml.opensmile


class AcousticsExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.tt_extractor = ml.feature_extraction.audio.tt_features_extractor.TTFeaturesExtractor(config)

    def extract(self, instance):
        duration = instance.audio.duration_seconds

        features = self.tt_extractor.extract(instance)
        features["ipu_duration"] = duration
        return features

    # def _dict_union_(self, *dicts):
    #     return dict(pair for d in dicts for pair in d.items())
    #
    # def batch_extract(self, instances):
    #     durations = [dict(ipu_duration=i.audio.duration_seconds) for i in instances]
    #     features1 = self.pitch_extractor.batch_extract(instances)
    #     features2 = self.intensity_extractor.batch_extract(instances)
    #     features3 = self.voice_quality_extractor.batch_extract(instances)
    #     return [self._dict_union_(durations[i], features1[i], features2[i], features3[i]) for i in range(0, len(instances))]
