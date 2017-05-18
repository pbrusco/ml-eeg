from .. import feature_extraction
import collections
import numpy as np
import scipy.stats
import ml.utils as utils
import ml.system as system
import ml.opensmile
import ml.parsing.arff


class PitchExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config, require=["pitch_config"])
        # self.voice_quality_config = self.params["voice_quality_config"]
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
        data = ml.opensmile.call_script(self.temp_folder, self.params["pitch_config"], instance.filename)

        times = ml.parsing.arff.get_column(data, "frameTime")
        pitch = ml.parsing.arff.get_column(data, "F0final_sma")

        # filter 0s
        times = times[pitch != 0]
        pitch = pitch[pitch != 0]

        return times, pitch
