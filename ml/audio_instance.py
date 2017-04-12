class AudioInstance:
    def __init__(self, filename, audio, speaker_gender):
        self.filename = filename
        self.audio = audio
        self.gender = speaker_gender
        assert self.gender.lower().strip() in ["m", "f"]

    def speaker_male(self):
        return self.gender.lower().strip() == "m"
