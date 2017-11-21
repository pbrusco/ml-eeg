from unittest import TestCase

import os
import ml.utils
import ml.feature_extraction.extractors
import ml.audio_instance
import ml.signal_processing
import ml.system

# ml.system.mute()

class TestAcousticExtractor(TestCase):
    def test_can_extract_from_simple_instance(self):
        config = dict(extraction_method = "acustics",
                        smile_extract_path="/home/pbrusco/apps/opensmile-2.3.0/bin/linux_x64_standalone_static/",
                        pitch_config="/home/pbrusco/apps/opensmile-2.3.0/config/smileF0.conf",
                        intensity_config="/home/pbrusco/apps/opensmile-2.3.0/config/intensity.conf",
                        voice_quality_config="/home/pbrusco/apps/opensmile-2.3.0/config/voice_quality.config",
                        extended_features=False,
                        extract_on_last_seconds=[0.2, 0.3, 0.5]
                     )
        extractor = ml.feature_extraction.extractors.create(config)
        here = os.path.dirname(os.path.abspath(__file__))
        filename = "{}/test.wav".format(here)
        audio = ml.signal_processing.load_wav(filename)
        instance = ml.audio_instance.AudioInstance(filename=filename, speaker_gender="M", audio=audio)
        features = extractor.extract(instance)

        self.assertAlmostEqual(features['f0_slope_last_200ms'], -172.45, delta=0.01)
        self.assertAlmostEqual(features['f0_slope_last_300ms'], -238.14, delta=0.01)
        self.assertAlmostEqual(features['f0_slope_last_500ms'], -19.20, delta=0.01)
        self.assertAlmostEqual(features['ipu_duration'], 2.33, delta=0.01)
        self.assertAlmostEqual(features['mean_intensity_last_200ms'], 65.14, delta=0.01)
        self.assertAlmostEqual(features['mean_intensity_last_300ms'], 65.24, delta=0.01)
        self.assertAlmostEqual(features['mean_intensity_last_500ms'], 63.62, delta=0.01)
        self.assertAlmostEqual(features['mean_jitter_last_200ms'], 0.01, delta=0.01)
        self.assertAlmostEqual(features['mean_jitter_last_300ms'], 0.01, delta=0.01)
        self.assertAlmostEqual(features['mean_jitter_last_500ms'], 0.01, delta=0.01)
        self.assertAlmostEqual(features['mean_nhr_last_200ms'], 0.19, delta=0.01)
        self.assertAlmostEqual(features['mean_nhr_last_300ms'], 0.15, delta=0.01)
        self.assertAlmostEqual(features['mean_nhr_last_500ms'], 0.18, delta=0.01)
        self.assertAlmostEqual(features['mean_pitch_last_200ms'], 159.95, delta=0.01)
        self.assertAlmostEqual(features['mean_pitch_last_300ms'], 170.05, delta=0.01)
        self.assertAlmostEqual(features['mean_pitch_last_500ms'], 168.00, delta=0.01)
        self.assertAlmostEqual(features['mean_shimmer_last_200ms'], 0.25, delta=0.01)
        self.assertAlmostEqual(features['mean_shimmer_last_300ms'], 0.08, delta=0.01)
        self.assertAlmostEqual(features['mean_shimmer_last_500ms'], 0.18, delta=0.01)

    def test_can_extract_from_multiple_instances(self):
        config = dict(extraction_method = "acustics",
                        smile_extract_path="/home/pbrusco/apps/opensmile-2.3.0/bin/linux_x64_standalone_static/",
                        pitch_config="/home/pbrusco/apps/opensmile-2.3.0/config/smileF0.conf",
                        intensity_config="/home/pbrusco/apps/opensmile-2.3.0/config/intensity.conf",
                        voice_quality_config="/home/pbrusco/apps/opensmile-2.3.0/config/voice_quality.config",
                        extended_features=False,
                        extract_on_last_seconds=[0.2, "all"]
                     )
        extractor = ml.feature_extraction.extractors.create(config)
        here = os.path.dirname(os.path.abspath(__file__))

        filename = "{}/test.wav".format(here)
        audio = ml.signal_processing.load_wav(filename)
        instance = ml.audio_instance.AudioInstance(filename=filename, speaker_gender="M", audio=audio)

        filename2 = "{}/test.wav".format(here)
        audio2 = ml.signal_processing.load_wav(filename2)
        instance2 = ml.audio_instance.AudioInstance(filename=filename2, speaker_gender="M", audio=audio2)

        all_features = extractor.batch_extract([instance, instance2])

        self.assertAlmostEqual(all_features[0]['f0_slope_last_200ms'], -172.45, delta=0.01)
        self.assertAlmostEqual(all_features[0]['ipu_duration'], 2.33, delta=0.01)
        self.assertAlmostEqual(all_features[0]['mean_intensity_last_200ms'], 65.14, delta=0.01)
        self.assertAlmostEqual(all_features[0]['mean_jitter_last_200ms'], 0.01, delta=0.01)
        self.assertAlmostEqual(all_features[0]['mean_nhr_last_200ms'], 0.19, delta=0.01)
        self.assertAlmostEqual(all_features[0]['mean_pitch_last_200ms'], 159.95, delta=0.01)
        self.assertAlmostEqual(all_features[0]['mean_shimmer_last_200ms'], 0.25, delta=0.01)
