from .. import feature_extraction
import collections
import numpy as np
import os.path
import inspect
import ml.utils as utils
import ml.system as system


class VoiceQualityExtractor(feature_extraction.FeatureExtractor):

    def __init__(self, config):
        self.params = utils.read_config(config)
        # self.voice_quality_config = self.params["voice_quality_config"]
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        last_seconds_values = self.params["extract_on_last_seconds"]

        duration = instance.audio.duration_seconds
        all_values = {}

        all_values["mean_shimmer"] = collections.defaultdict(lambda: np.nan)
        all_values["mean_jitter"] = collections.defaultdict(lambda: np.nan)
        all_values["mean_nhr"] = collections.defaultdict(lambda: np.nan)

        for last_secs in last_seconds_values:
            if last_secs == "all":
                last_secs = duration
                label = "all"
            else:
                label = last_secs

            if duration < last_secs:
                continue

            jitter, shimmer, nhr = self.extract_voice_quality(instance, start=duration - last_secs, end=duration)
            all_values["mean_jitter"][label] = jitter
            all_values["mean_shimmer"][label] = shimmer
            all_values["mean_nhr"][label] = nhr

        feat = {}

        for last_secs in last_seconds_values:
            if last_secs == "all":
                in_ms = "all"
            else:
                in_ms = "last_{}ms".format(int(last_secs * 1000))

            for feat_name in ["mean_jitter", "mean_shimmer", "mean_nhr"]:
                feat["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][last_secs]

        return feat

    def extract_voice_quality(self, instance, start, end):
        min_pitch = 50 if instance.speaker_male() else 75
        max_pitch = 300 if instance.speaker_male() else 500

        script_path = os.path.dirname(inspect.getfile(inspect.currentframe())) + "/voice-analysis.praat"
        try:
            output = system.run_external_command("praat {}".format(script_path), non_named_params=[instance.filename, start, end, min_pitch, max_pitch])
        except:
            system.warning("Check this case (Extracting voice quality error)!!!!")  # TODO: fix
            return np.nan, np.nan, np.nan

        values = dict([o.split(":") for o in output.split()])

        jitter = float(values["sound_voiced_local_jitter"]) if "undefined" not in values["sound_voiced_local_jitter"] else np.nan
        shimmer = float(values["sound_voiced_local_shimmer"]) if "undefined" not in values["sound_voiced_local_shimmer"] else np.nan
        nhr = float(values["noise_to_harmonics_ratio"]) if "undefined" not in values["noise_to_harmonics_ratio"] else np.nan

        return jitter, shimmer, nhr

    def batch_extract(self, instances):
        return [self.extract(instance) for instance in instances]
