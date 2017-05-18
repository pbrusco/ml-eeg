# coding: utf-8
from .. import feature_extraction
import ml.feature_extraction.audio.pitch_extractor
import ml.feature_extraction.audio.intensity_extractor
import ml.feature_extraction.audio.voice_quality_extractor

import ml.parsing.arff
import ml.opensmile


class AcousticsExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.pitch_extractor = ml.feature_extraction.audio.pitch_extractor.PitchExtractor(config)
        self.intensity_extractor = ml.feature_extraction.audio.intensity_extractor.IntensityExtractor(config)
        self.voice_quality_extractor = ml.feature_extraction.audio.voice_quality_extractor.VoiceQualityExtractor(config)

    def extract(self, instance):
        duration = instance.audio.duration_seconds

        feat = dict(ipu_duration=duration)
        feat.update(self.pitch_extractor.extract(instance))
        feat.update(self.intensity_extractor.extract(instance))
        feat.update(self.voice_quality_extractor.extract(instance))
        return feat
