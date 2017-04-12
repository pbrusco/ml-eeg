# coding: utf-8
import feature_extraction
import collections
import lib.system
import arff
import numpy as np
import scipy.stats
import time
import os.path
import inspect


class AudioExtractor(feature_extraction.FeatureExtractor):
    pass


class AcousticsExtractor(AudioExtractor):
    # ; Duration
    # ; Entonación (F0 slope)
    # ; Intensity Level (Mean Intensity)
    # ; Pitch Level (Mean Pitch)
    # ; Voice Quality (Jitter)
    # ; Voice Quality (Shimmer)
    # ; Voice Quality (NHR)

    def __init__(self, params):
        self.pitch_config = params["pitch_config"]
        self.intensity_config = params["intensity_config"]
        # self.voice_quality_config = params["voice_quality_config"]
        self.temp_folder = "/tmp/opensmile_arffs/"
        self.extended_features = params["extended_features"]
        self.extract_on_last_seconds = params["extract_on_last_seconds"]
        lib.system.mkdir_p(self.temp_folder)

    def extract(self, instance):
        last_seconds_values = self.extract_on_last_seconds

        duration = instance.audio.duration_seconds
        times_pitch, pitch = self.extract_pitch(instance)
        times_intensities, intensities = self.extract_intensity(instance)

        all_values = {}

        all_values["f0_slope"] = collections.defaultdict(lambda: np.nan)

        all_values["mean_pitch"] = collections.defaultdict(lambda: np.nan)
        all_values["mean_intensity"] = collections.defaultdict(lambda: np.nan)
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

            for times, values, feat_name in zip(
                [times_pitch, times_intensities],
                [pitch, intensities],
                ["pitch", "intensity"]
            ):

                indices = (times >= duration - last_secs)
                all_values["mean_" + feat_name][label] = np.mean(values[indices])

            indices = (times_pitch >= duration - last_secs)
            if len(times_pitch[indices]) > 3:
                all_values["f0_slope"][label] = scipy.stats.linregress(times_pitch[indices], pitch[indices])[0]  # [0] => slope (m)

        feat = dict(ipu_duration=duration)

        for last_secs in last_seconds_values:
            if last_secs == "all":
                in_ms = "all"
            else:
                in_ms = "last_{}ms".format(int(last_secs * 1000))

            for feat_name in ["f0_slope", "mean_pitch", "mean_intensity", "mean_jitter", "mean_shimmer", "mean_nhr"]:
                feat["{}_{}".format(feat_name, in_ms)] = all_values[feat_name][last_secs]

        if self.extended_features:
            feat["pitch"] = pitch
            feat["times_pitch"] = times_pitch

            feat["intensities"] = intensities
            feat["times_intensities"] = times_intensities

        return feat

    def extract_pitch(self, instance):
        data = self.call_opensmile_script(self.pitch_config, instance.filename)

        times = self.get_column(data, "frameTime")
        pitch = self.get_column(data, "F0final_sma")

        # filter 0s
        times = times[pitch != 0]
        pitch = pitch[pitch != 0]

        return times, pitch

    def extract_intensity(self, instance):
        data = self.call_opensmile_script(self.intensity_config, instance.filename)

        times = self.get_column(data, "frameTime")
        intensities = self.get_column(data, "pcm_intensity_sma")

        times = times[intensities != 0]
        intensities = intensities[intensities != 0]
        intensities_in_dbs = 10 * np.log10(intensities / 10e-14)  # TODO: revisar using I0 = 10 −6 as defined in the opensmile book
        return times, intensities_in_dbs

    def extract_voice_quality(self, instance, start, end):
        min_pitch = 50 if instance.speaker_male() else 75
        max_pitch = 300 if instance.speaker_male() else 500

        script_path = os.path.dirname(inspect.getfile(inspect.currentframe())) + "/voice-analysis.praat"
        try:
            script_path = os.path.dirname(inspect.getfile(inspect.currentframe())) + "/voice-analysis-voiced.praat"
            output = lib.system.run_external_command("praat {}".format(script_path), non_named_params=[instance.filename, start, end, min_pitch, max_pitch])
            values = dict([o.split(":") for o in output.split()])
            jitter = float(values["sound_voiced_local_jitter"]) if "undefined" not in values["sound_voiced_local_jitter"] else np.nan
            shimmer = float(values["sound_voiced_local_shimmer"]) if "undefined" not in values["sound_voiced_local_shimmer"] else np.nan
            nhr = float(values["noise_to_harmonics_ratio"]) if "undefined" not in values["noise_to_harmonics_ratio"] else np.nan
        except:
            print "No voiced frames on range {}-{}".format(start, end)
            jitter = np.nan
            shimmer = np.nan
            nhr = np.nan

        return jitter, shimmer, nhr

        # data = self.call_opensmile_script(self.voice_quality_config, instance.filename)
        # times = self.get_column(data, "frameTime")

        # jitter = self.get_column(data, "jitterLocal")
        # times_jitter = times[jitter != 0]
        # jitter = jitter[jitter != 0]
        #
        # shimmer = self.get_column(data, "shimmerLocal")
        # times_shimmer = times[shimmer != 0]
        # shimmer = shimmer[shimmer != 0]
        #
        # nhr = self.get_column(data, "logHNR")
        # times_nhr = times[nhr != 0]
        # nhr = nhr[nhr != 0]

        # return times_jitter, jitter, times_shimmer, shimmer, times_nhr, nhr

    def call_opensmile_script(self, config, filename):
        timestamp = time.time()
        temp_output = "{}/{}{}.arff".format(self.temp_folder, timestamp, os.path.basename(filename))
        data = {}
        lib.system.run_command("SMILExtract -C {} -I {} -l 0 -arffoutput {} -appendarff 0".format(config, filename, temp_output))
        data = arff.load(open(temp_output, 'rb'))
        lib.system.rm(temp_output)
        return data

    def get_column(self, data, column_name):
        column_id = [a[0] for a in data["attributes"]].index(column_name)
        values = np.array([float(d[column_id]) for d in data["data"]])
        return values
