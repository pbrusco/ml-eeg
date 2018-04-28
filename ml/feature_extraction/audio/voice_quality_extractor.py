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

class VoiceQualityExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config, require=["voice_quality_config"])
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        # last_seconds_values = self.params["extract_on_last_seconds"]
        windows = self.params["extract_on_windows"]

        duration = instance.audio.duration_seconds
        times, jitter, shimmer, logHNR = self.vq_track(instance)

        times = times - times.max()

        feat = {}

        if self.params["extended_features"]:
            feat["times_logHNR"] = times
            feat["times_shimmer"] = times[shimmer > 0]
            feat["times_jitter"] = times[jitter > 0]

            feat["logHNR"] = logHNR
            feat["shimmer"] = shimmer[shimmer > 0]
            feat["jitter"] = jitter[jitter > 0]

        values = dict(jitter=jitter, shimmer=shimmer, logHNR=logHNR)


        for klass in ["jitter", "shimmer", "logHNR"]:
            all_values = {}

            all_values["mean_{}".format(klass)] = collections.defaultdict(lambda: np.nan)
            all_values["{}_slope".format(klass)] = collections.defaultdict(lambda: np.nan)

            for (w_from, w_to) in windows:
                window = (w_from, w_to)
                if not w_from:
                    indices = np.array([True]*len(times))
                else:
                    indices = (times >= w_from) & (times < w_to)

                all_values["mean_{}".format(klass)][window] = np.mean(values[klass][indices])

                if sum(indices) > 5: #suficientes valores para calcular slope
                    all_values["{}_slope".format(klass)][window] = scipy.stats.linregress(times[indices], values[klass][indices])[0]  # [0] => slope (m)

            for (w_from, w_to) in windows:
                window = (w_from, w_to)
                if not w_from:
                    in_ms = "all_ipu"
                else:
                    in_ms = "({},{})".format(int(w_from*1000), int(w_to*1000))

                for feat_name in ["{}_slope".format(klass), "mean_{}".format(klass)]:
                    feat["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][window]

        return feat


    def vq_track(self, instance):
        data = ml.opensmile.call_script(self.params["smile_extract_path"], self.temp_folder, self.params["voice_quality_config"], instance.filename)

        times = ml.parsing.arff.get_column(data, "frameTime")
        jitter = ml.parsing.arff.get_column(data, "jitterLocal")
        shimmer = ml.parsing.arff.get_column(data, "shimmerLocal")
        logHNR = ml.parsing.arff.get_column(data, "logHNR")

        return times, jitter, shimmer, logHNR


    def batch_extract(self, instances):
        return [self.extract(instance) for instance in instances]
