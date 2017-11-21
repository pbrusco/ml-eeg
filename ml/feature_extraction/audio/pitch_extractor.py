from .. import feature_extraction
import collections
import pysptk
import numpy as np
import scipy.stats
import ml.utils as utils
import ml.system as system
import ml.opensmile
import ml.parsing.arff
import math


class PitchExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config, require=["pitch_config"])
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        last_seconds_values = self.params["extract_on_last_seconds"]

        duration = instance.audio.duration_seconds
        times_pitch, pitch = self.pitch_track(instance)

        all_values = {}

        all_values["f0_slope"] = collections.defaultdict(lambda: np.nan)
        all_values["mean_pitch"] = collections.defaultdict(lambda: np.nan)

        for last_secs in last_seconds_values:
            if last_secs == "all":
                last_secs = duration
                label = "all"
            else:
                label = last_secs

            if duration < last_secs:
                continue

            indices = (times_pitch >= duration - last_secs)
            all_values["mean_pitch"][label] = np.mean(pitch[indices])

            indices = (times_pitch >= duration - last_secs)
            if len(times_pitch[indices]) > 3:
                all_values["f0_slope"][label] = scipy.stats.linregress(times_pitch[indices], pitch[indices])[0]  # [0] => slope (m)

        feat = {}
        for last_secs in last_seconds_values:
            if last_secs == "all":
                in_ms = "all"
            else:
                in_ms = "last_{}ms".format(int(last_secs * 1000))

            for feat_name in ["f0_slope", "mean_pitch"]:
                feat["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][last_secs]

        if self.params["extended_features"]:
            feat["pitch"] = pitch
            feat["times_pitch"] = times_pitch

        return feat

    def pitch_track(self, instance):
        data = ml.opensmile.call_script(self.params["smile_extract_path"], self.temp_folder, self.params["pitch_config"], instance.filename)

        times = ml.parsing.arff.get_column(data, "frameTime")
        pitch = ml.parsing.arff.get_column(data, "F0final_sma")

        times = times[pitch > 0]
        pitch = pitch[pitch > 0]

        return times, pitch
        #
        # fs = instance.audio.frame_rate
        # ms_10_in_frames = math.ceil(fs / 100) # Extract every 10 ms.
        # min_pitch = 50 if instance.speaker_male() else 75
        # max_pitch = 300 if instance.speaker_male() else 500
        # x = np.array(instance.audio.get_array_of_samples())
        # f0_swipe = pysptk.swipe(x.astype(np.float64), fs=fs, hopsize=ms_10_in_frames, min=min_pitch, max=max_pitch, otype="f0", threshold=0.27)
        #
        # times = np.arange(0, len(f0_swipe)) / 100.0
        # filtered_times = times[f0_swipe > 0]
        # filtered_f0_swipe = f0_swipe[f0_swipe > 0]
        #
        # return filtered_times, filtered_f0_swipe

    def batch_extract(self, instances):
        return [self.extract(instance) for instance in instances]
