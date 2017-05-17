# coding=utf-8

from .equality import EqualityMixin


class Turn(EqualityMixin):
    def __init__(self, start, end, intervals):
        self.start = float(start)
        self.end = float(end)
        self.duration = self.end - self.start
        self.intervals = intervals

    def __repr__(self):
        return "T {a} to {b} with: {c} \n".format(a=self.start, b=self.end, c=self.intervals)


def next_ipu(vad):
    return vad.next("1")


def next_silence(vad):
    return vad.next("0")


def is_ipu(interval):
    return interval.value == "1"


def is_silence(interval):
    return interval.value == "0"


def intervals_between(t0, t1, vad):

    return [interval for interval in vad if interval.start <= t1 and interval.end >= t0]


def next_turn(vad, speaker2_vad):
    intervals = []

    following_ipu = next_ipu(vad)
    if not following_ipu:
        return None

    start = following_ipu.start
    intervals.append(following_ipu)

    while True:
        if not vad.has_next():
            return Turn(start, vad.current().end, intervals)

        next_interval = next(vad)

        if is_silence(next_interval):
            overlaped_intervals = intervals_between(next_interval.start, next_interval.end, speaker2_vad)

            if any(map(is_ipu, overlaped_intervals)):
                end = next_interval.start
                return Turn(start, end, intervals)

        intervals.append(next_interval)


def turns(vad, speaker2_vad):
    turns = []
    while True:
        turn = next_turn(vad, speaker2_vad)
        if not turn:
            return turns
        else:
            turns.append(turn)
