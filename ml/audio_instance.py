# coding: utf-8
from . import system


class AudioInstance:
    def __init__(self, filename, audio, speaker_gender):
        self.filename = filename
        self.audio = audio
        self.gender = speaker_gender.lower().strip()
        assert system.exists(filename), "filename {} does not exist"
        assert self.gender in ["m", "f"], "gender options are 'm' or 'f'"

    def speaker_male(self):
        return self.gender == "m"
