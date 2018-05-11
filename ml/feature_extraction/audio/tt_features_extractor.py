from .. import feature_extraction
import collections
import numpy as np
import scipy.stats
import ml.utils as utils
import ml.system as system
import ml.opensmile
import ml.parsing.arff


class TTFeaturesExtractor(feature_extraction.FeatureExtractor):
    def __init__(self, config):
        self.params = utils.read_config(config, require=["opensmile_config"])
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        windows = self.params["extract_on_windows"]

        times, series = self.track(instance)

        res = {}
        if len(times) == 0:
            raise Exception("OpenSmile extraction error with file {} (dur={})(skipping file)".format(instance.filename, instance.audio.duration_seconds))

        times = times - times.max()

        for feature, values in series.items():
            times_feature = times[~np.isnan(values)]
            feature_values = values[~np.isnan(values)]

            if self.params["extended_features"]:
                res["times_{}".format(feature)] = times_feature
                res[feature] = feature_values

            all_values = {}

            all_values["mean_{}".format(feature)] = collections.defaultdict(lambda: np.nan)
            all_values["{}_slope".format(feature)] = collections.defaultdict(lambda: np.nan)

            for (w_from, w_to) in windows:
                window = (w_from, w_to)
                if not w_from:
                    indices = np.array([True] * len(times_feature))
                else:
                    indices = (times_feature >= w_from) & (times_feature < w_to)

                all_values["mean_{}".format(feature)][window] = np.mean(feature_values[indices])
                if sum(indices) > 2:  # suficientes valores para calcular slope
                    all_values["{}_slope".format(feature)][window] = scipy.stats.linregress(times_feature[indices], feature_values[indices])[0]  # [0] => slope (m)

            for (w_from, w_to) in windows:
                window = (w_from, w_to)
                if not w_from:
                    in_ms = "all_ipu"
                else:
                    in_ms = "({},{})".format(int(w_from * 1000), int(w_to * 1000))

                for feat_name in ["{}_slope".format(feature), "mean_{}".format(feature)]:
                    res["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][window]
        return res

    def track(self, instance):
        data = ml.opensmile.call_script(self.params["smile_extract_path"], self.temp_folder, self.params["opensmile_config"], instance.filename)
        series = {}
        times = ml.parsing.arff.get_column(data, "frameTime")

        pitch = ml.parsing.arff.get_column(data, "F0final_sma")
        intensity = ml.parsing.arff.get_column(data, "pcm_intensity_sma")
        jitter = ml.parsing.arff.get_column(data, "jitterLocal")
        shimmer = ml.parsing.arff.get_column(data, "shimmerLocal")
        logHNR = ml.parsing.arff.get_column(data, "logHNR")

        intensity[~(intensity > 0)] = np.nan
        jitter[~(pitch > 0)] = np.nan
        shimmer[~(pitch > 0)] = np.nan
        logHNR[~(pitch > 0)] = np.nan
        pitch[~(pitch > 0)] = np.nan

        series["pitch"] = pitch
        series["intensity"] = intensity
        series["jitter"] = jitter
        series["shimmer"] = shimmer
        series["logHNR"] = logHNR

        return times, series

    def batch_extract(self, instances):
        return [self.extract(instance) for instance in instances]
