# coding=utf-8

from . import interval
import ml.system as system


class Intervals():
    def __init__(self, intervals):
        if len(intervals) > 0 and (type(intervals[0]) is tuple or type(intervals[0]) is list):
            intervals = [interval.from_tuple(t) for t in intervals]

        self.intervals = intervals
        self.__reset__()

    def __reset__(self):
        self.counter = 0

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and self.intervals == other.intervals)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        for i in self.intervals:
            yield i

    def __getitem__(self, idx):
        return list(self)[idx]

    def __len__(self):
        return len(self.intervals)

    def diff(self, x):
        differences = []
        if len(self.intervals) != len(x.intervals):
            system.warning("number of intervals: {} vs {}".format(len(self.intervals), len(x.intervals)))

        for i in x:
            if i.start != self.current().start:
                system.error("different starts on interval ({}) {} vs {}".format(self.counter + 1, i.start, self.current().start))
                raise Exception("Intervals are not the same")

            if i.value != self.current().value:
                differences.append((self.counter + 1, i.start, i.end, i.value, self.current().value))
            next(self)

        self.__reset__()

        return differences

    def merge(self, x):
        if len(self.intervals) != len(x.intervals):
            system.warning("number of intervals: {} vs {}".format(len(self.intervals), len(x.intervals)))

        res = []

        for i in x:
            if i.start != self.current().start:
                system.error("different starts on interval ({}) {} vs {}".format(self.counter + 1, i.start, self.current().start))
                raise Exception("Intervals are not the same")

            if i.value == self.current().value:
                res.append(i)
            else:
                res.append(interval.Interval(i.start, i.end, "A"))
                
            next(self)

        self.__reset__()
        return Intervals(res)

    def current(self):
        return self.intervals[self.counter]

    def next(self, value=None):
        if self.has_next():
            self.counter += 1
            val = self.intervals[self.counter]
            if value and value != val.value:
                return self.next(value)
            else:
                return val
        else:
            return None

    def values_in_range(self, start, end):
        values = []
        for i in self.intervals:
            if i.end < start:
                continue
            elif i.start > end:
                break
            else:
                values.append(i.value)
        return values

    def next_until(self, time):
        if self.current().end < time and self.has_next():
            next(self)
            return self.next_until(time)
        else:
            return self.current()

    def has_next(self):
        return self.counter < len(self.intervals) - 1

    def __repr__(self):
        return "Wavesurfer: " + str(self.intervals)

    def as_vad(self):
        vad_intervals = []
        current_ipu = None
        for i in self:
            if i.value == "#":
                if current_ipu:
                    vad_intervals.append(current_ipu)
                    current_ipu = None
                vad_intervals.append(interval.Interval(i.start, i.end, "0"))

            else:
                if not current_ipu:
                    current_ipu = interval.Interval(i.start, i.end, "1")
                else:
                    current_ipu = interval.Interval(current_ipu.start, i.end, "1")

        return Intervals(vad_intervals)

    def as_list(self):
        return [i.as_tuple() for i in self]
