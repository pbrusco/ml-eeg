import system
from . import interval
from . import wavesurfer


class ParseError(Exception):
    def __init__(self, value):
        self.value = value


class WaveParser:
    def __init__(self, wavesurfer_lines):
        self.lastEnd = 0
        self.lines = wavesurfer_lines

    def parse(self):
        values = [x.split() for x in self.lines]

        def as_interval(y):
            return interval.Interval(y[0], y[1], y[2])

        intervals = []
        for v in values:
            i = as_interval(v)
            if i.start != self.lastEnd or i.end < self.lastEnd:
                raise ParseError("Malformed Wavesurfer, check for missing time values")
            else:
                self.lastEnd = i.end

            intervals.append(interval)

        return wavesurfer.WaveSurfer(intervals)


def read(wavesurfer_file):
    # Expects a file with the: "time0 time1 value\n", format, for example:
    # 0.00 0.52 Hello
    # 0.52 1.12 Bye
    # Returns a list of intervals: [I(t0, t1, val1), I(t1, t2, val2), ...]
    # Time values are parsed as floats.
    if not system.exists(wavesurfer_file):
        raise Exception("Missing file: " + wavesurfer_file)

    with open(wavesurfer_file, "r") as file:
        lines = file.readlines()

    return WaveParser(lines).parse()
