from .. import feature_extraction
import collections
import numpy as np
import ml.utils as utils
import ml.system as system
import ml.opensmile
import ml.parsing.arff


class IntensityExtractor(feature_extraction.FeatureExtractor):

    def __init__(self, config):
        self.params = utils.read_config(config, require=["intensity_config"])
        # self.voice_quality_config = self.params["voice_quality_config"]
        self.temp_folder = config["temp_folder"] if "temp_folder" in config else "/tmp/opensmile_arffs/"
        system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        last_seconds_values = self.params["extract_on_last_seconds"]

        duration = instance.audio.duration_seconds
        times_intensities, intensities = self.intensity_track(instance)

        all_values = {}

        all_values["mean_intensity"] = collections.defaultdict(lambda: np.nan)

        for last_secs in last_seconds_values:
            if last_secs == "all":
                last_secs = duration
                label = "all"
            else:
                label = last_secs

            if duration < last_secs:
                continue

            indices = (times_intensities >= duration - last_secs)
            all_values["mean_intensity"][label] = np.mean(intensities[indices])

        feat = {}

        for last_secs in last_seconds_values:
            if last_secs == "all":
                in_ms = "all"
            else:
                in_ms = "last_{}ms".format(int(last_secs * 1000))

            feat["mean_intensity_{}".format(in_ms)] = all_values["mean_intensity"][last_secs]

        if self.params["extended_features"]:
            feat["intensities"] = intensities
            feat["times_intensities"] = times_intensities

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
