from eeg_feature_extractor import EEGFeatureExtractor
import numpy as np
import ml.data_import as data_import
import ml.signal_processing as signal_processing


class WindowedExtractor(EEGFeatureExtractor):
    def __init__(self, config):
        super(WindowedExtractor, self).__init__(config)
        self.window_sizes = self.params["window_sizes"]
        self.step_in_frames = self.params["step_in_frames"]

    def extract(self, trial):
        # Trial shape: (channels x samples)
        assert self.n_channels == trial.shape[0], "the number of channels in the extractor configuration doen't match {} vs {}".format(self.n_channels, trial.shape[0])
        assert self.n_samples == trial.shape[1], "the number of samples in the extractor configuration doen't match {} vs {}".format(self.n_samples, trial.shape[1])

        features = []
        for window_size_in_secs in self.window_sizes:
            ws_frames = int(window_size_in_secs * self.sfreq)
            if self.verbose:
                print("ws frames: ", ws_frames)
            for starting_sample, chunk in signal_processing.sliding_window(trial,
                                                                           window_size_in_frames=ws_frames,
                                                                           step_in_frames=self.step_in_frames,
                                                                           time_limits=self.extraction_time_limits,
                                                                           t0=self.trial_tmin,
                                                                           freq=self.sfreq,
                                                                           debugging=self.verbose,
                                                                           ):
                window_coefficients = []
                for ch in self.channels_to_extract_from:
                    y = chunk[ch, :]
                    window_coefficients.append(y.mean())
                features.extend(window_coefficients)

        features = np.array(features)
        return features

    def mapping(self):
        i = 0
        dummy_epoch = np.zeros((self.n_channels, self.n_samples))  # (channels x samples)
        montage = data_import.read_montage(self.params["montage"])

        res = {}
        for window_size_in_secs in self.window_sizes:
            ws_frames = int(window_size_in_secs * self.sfreq)
            for starting_sample, chunk in signal_processing.sliding_window(dummy_epoch,
                                                                           window_size_in_frames=ws_frames,
                                                                           step_in_frames=self.step_in_frames,
                                                                           time_limits=self.extraction_time_limits,
                                                                           t0=self.trial_tmin,
                                                                           freq=self.sfreq,
                                                                           ):
                for ch in self.channels_to_extract_from:
                    t0, t1 = signal_processing.frame_time_limits(starting_sample, self.sfreq, self.trial_tmin, ws_frames)
                    res[i] = dict(
                        channel=ch,
                        channel_name=montage.ch_names[ch],
                        starting_sample=starting_sample,
                        window_size=window_size_in_secs,
                        starting_time=round(t0, 3),
                        end_time=round(t1, 3),
                        channel_position=montage.pos[ch],
                        name="(w{}|c{}|{}|{})".format(round(window_size_in_secs, 2), ch, round(t0, 2), round(t1, 2))
                    )
                    i = i + 1
        return res

    def feature_names(self):
        return [c["name"] for c in self.mapping().values()]
