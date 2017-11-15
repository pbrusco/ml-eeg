from .eeg_feature_extractor import EEGFeatureExtractor

import numpy as np
import ml.data_import as data_import

from scipy import signal


class FreqExtractor(EEGFeatureExtractor):
    def __init__(self, config):
        super(FreqExtractor, self).__init__(config)
        self.min_freq = self.params["min_freq"]
        self.max_freq = self.params["max_freq"]

    def extract(self, trial):
        assert self.n_channels == trial.shape[0], "the number of channels in the extractor configuration doen't match {} vs {}".format(self.n_channels, trial.shape[0])
        assert self.n_samples == trial.shape[1], "the number of samples in the extractor configuration doen't match {} vs {}".format(self.n_samples, trial.shape[1])
        ti, tf = self.extraction_time_limits

        initial_frame = int((ti - self.trial_tmin) * self.sfreq)
        end_frame = int((tf - self.trial_tmin) * self.sfreq)
        data = trial[self.channels_to_extract_from, initial_frame:end_frame]

        freqs_by_channel = []

        for ch in self.channels_to_extract_from:
            x = data[ch, :]
            freqs, Pxx = signal.welch(x, fs=self.sfreq)
            idx_30hz = np.argmax(freqs > self.max_freq)  # No more than x hz
            idx_1hz = np.argmin(freqs < self.min_freq)  # No less than y Hz

            freqs = freqs[idx_1hz:idx_30hz]
            presence_of_freq = Pxx[idx_1hz:idx_30hz]

            freqs_by_channel.extend(presence_of_freq)
        res = np.array(freqs_by_channel)
        return res

    def mapping(self):
        trial = np.zeros((self.n_channels, self.n_samples))  # (channels x samples)
        montage = data_import.read_montage('/home/pbrusco/projects/montages/LNI.sfp')

        ti, tf = self.extraction_time_limits

        initial_frame = int((ti - self.trial_tmin) * self.sfreq)
        end_frame = int((tf - self.trial_tmin) * self.sfreq)
        data = trial[self.channels_to_extract_from, initial_frame:end_frame]
        res = {}

        i = 0
        for ch in self.channels_to_extract_from:
            x = data[ch, :]
            freqs, Pxx = signal.welch(x, fs=self.sfreq)
            idx_30hz = np.argmax(freqs > self.max_freq)  # No more than x hz
            idx_1hz = np.argmin(freqs < self.min_freq)  # No less than y Hz

            freqs = freqs[idx_1hz:idx_30hz]
            for freq in freqs:
                res[i] = dict(channel=ch,
                              channel_name=montage.ch_names[ch],
                              freq=freq,
                              channel_position=montage.pos[ch],
                              name="c{}|{}Hz)".format(ch, round(freq, 2))
                              )
                i = i + 1

        return res

    def feature_names(self):
        return [c["name"] for c in list(self.mapping().values())]
