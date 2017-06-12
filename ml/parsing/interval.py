# coding=utf-8

from .equality import EqualityMixin


class Interval(EqualityMixin):
    def __init__(self, start, end, value):
        self.start = float(start)
        self.end = float(end)
        self.value = value
        self.duration = self.end - self.start

    def __iter__(self):
        yield (self.start, self.end, self.value)

    def __repr__(self):
        return "I({a}, {b}, {c})".format(a=self.start, b=self.end, c=self.value)

    def as_tuple(self):
        return (self.start, self.end, self.value)


def from_tuple(t):
    return Interval(t[0], t[1], t[2])
