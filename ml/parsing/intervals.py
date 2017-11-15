# coding=utf-8

import ml.system as system
from . import textgrid


class Intervals():
    def __init__(self, intervals):
        if len(intervals) > 0 and (type(intervals[0]) is tuple or type(intervals[0]) is list):
            intervals = [textgrid.interval(t[0], t[1], t[2]) for t in intervals]

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

    def validate_values(self, values):
        warnings = []
        for idx, i in enumerate(self.intervals):
            if i.mark not in values:
                warnings.append("(id: {}, time: {}) invalid label \"{}\"".format(idx + 1, i.minTime, i.mark))
        return warnings

    def validate_for_diff(self, x):
        warnings = []
        if len(self.intervals) != len(x.intervals):
            warnings.append("number of intervals: {} vs {}".format(len(self.intervals), len(x.intervals)))
        for i in x:
            if i.minTime != self.current().minTime:
                warnings.append("different starts on interval ({}) {} vs {}".format(self.counter + 1, i.minTime, self.current().minTime))
                break
            next(self)

        self.__reset__()

        return warnings

    def diff(self, x):
        differences = []
        if len(self.intervals) != len(x.intervals):
            system.warning("number of intervals: {} vs {}".format(len(self.intervals), len(x.intervals)))

        for i in x:
            if i.minTime != self.current().minTime:
                system.warning("different starts on interval ({}) {} vs {}".format(self.counter + 1, i.minTime, self.current().minTime))
                break
            if i.mark.strip() != self.current().mark.strip():
                differences.append((self.counter + 1, i.minTime, i.maxTime, i.mark, self.current().mark))
            elif i.mark != self.current().mark:
                system.warning("Labels '{}' vs '{}' ignored from diff".format(i.mark, self.current().mark))

            next(self)

        self.__reset__()

        return differences

    def merge(self, x):
        if len(self.intervals) != len(x.intervals):
            system.warning("number of intervals: {} vs {}".format(len(self.intervals), len(x.intervals)))

        res = []

        for i in x:
            if i.minTime != self.current().minTime:
                system.error("different starts on interval ({}) {} vs {}".format(self.counter + 1, i.minTime, self.current().minTime))
                raise Exception("Intervals are not the same")

            if i.mark == self.current().mark:
                res.append(i)
            else:
                res.append(textgrid.Interval(i.minTime, i.maxTime, "A"))

            next(self)

        self.__reset__()
        return Intervals(res)

    def current(self):
        return self.intervals[self.counter]

    def next(self, value=None):
        if self.has_next():
            self.counter += 1
            val = self.intervals[self.counter]
            if value and value != val.mark:
                return self.next(value)
            else:
                return val
        else:
            return None

    def values_in_range(self, start, end):
        values = []
        for i in self.intervals:
            if i.maxTime < start:
                continue
            elif i.minTime > end:
                break
            else:
                values.append(i.mark)
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
            if i.mark == "#":
                if current_ipu:
                    vad_intervals.append(current_ipu)
                    current_ipu = None
                vad_intervals.append(textgrid.Interval(i.minTime, i.maxTime, "0"))

            else:
                if not current_ipu:
                    current_ipu = textgrid.Interval(i.minTime, i.maxTime, "1")
                else:
                    current_ipu = textgrid.Interval(current_ipu.minTime, i.maxTime, "1")

        return Intervals(vad_intervals)

    def as_list(self):
        return [(i.minTime, i.maxTime, i.mark) for i in self]
