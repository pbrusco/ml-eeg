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


class IntensityExtractor(feature_extraction.FeatureExtractor):

    def __init__(self, config):
        self.params = utils.read_config(config, require=["intensity_config"])
        # self.voice_quality_config = self.params["voice_quality_config"]
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        # last_seconds_values = self.params["extract_on_last_seconds"]
        windows = self.params["extract_on_windows"]

        duration = instance.audio.duration_seconds
        times_intensity, intensity = self.intensity_track(instance)

        all_values = {}

        all_values["intensity_slope"] = collections.defaultdict(lambda: np.nan)
        all_values["mean_intensity"] = collections.defaultdict(lambda: np.nan)

        times_intensity = times_intensity - times_intensity.max() # Alineando a 0.

        for (w_from, w_to) in windows:
            window = (w_from, w_to)
            if not w_from:
                indices = np.array([True]*len(times_intensity))
            else:
                indices = (times_intensity >= w_from) & (times_intensity < w_to)

            all_values["mean_intensity"][window] = np.mean(intensity[indices])

            if sum(indices) > 5: #suficientes valores para calcular slope
                all_values["intensity_slope"][window] = scipy.stats.linregress(times_intensity[indices], intensity[indices])[0]  # [0] => slope (m)

        feat = {}
        for (w_from, w_to) in windows:
            window = (w_from, w_to)
            if not w_from:
                in_ms = "all_ipu"
            else:
                in_ms = "({},{})".format(int(w_from*1000), int(w_to*1000))

            for feat_name in ["intensity_slope", "mean_intensity"]:
                feat["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][window]

        if self.params["extended_features"]:
            feat["intensity"] = intensity
            feat["times_intensity"] = times_intensity

        return feat

    def intensity_track(self, instance):
        data = ml.opensmile.call_script(self.params["smile_extract_path"], self.temp_folder, self.params["intensity_config"], instance.filename)

        times = ml.parsing.arff.get_column(data, "frameTime")
        intensities = ml.parsing.arff.get_column(data, "pcm_intensity_sma")

        times = times[intensities != 0]
        intensities = intensities[intensities != 0]
        intensities_in_dbs = 10 * np.log10(intensities / 10e-14)  # TODO: revisar using I0 = 10 −6 as defined in the opensmile book
        return times, intensities_in_dbs

    def batch_extract(self, instances):
        return [self.extract(instance) for instance in instances]
