# coding: utf-8
from . import system
from . import signal_processing


class AudioInstance:
    def __init__(self, filename, speaker_gender):
        self.filename = filename
        self.audio = signal_processing.load_wav(filename)
        self.gender = speaker_gender.lower().strip()
        assert system.exists(filename), "filename {} does not exist"
        assert self.gender in ["m", "f"], "gender options are 'm' or 'f'"

    def speaker_male(self):
        return self.gender == "m"
