import lib.system
import common
from interval import Interval
from wavesurfer import WaveSurfer
import re
from IPython import embed
import codecs


def try_getting(key, line):
    if line.startswith("{} =".format(key)):
        return line.split("=")[1].strip().replace("\"", "")
    else:
        return None


def get(key, line):
    assert line.startswith("{} =".format(key)), "key: {}, not found in: {}".format(key, line)
    k, v = [a.strip() for a in line.split("=")]
    return v.replace("\"", "")


def assert_is(key, line):
    assert line.startswith(key), "line didnt start with: {} ({})".format(key, line)


class ParseError(Exception):
    def __init__(self, value):
        self.value = value


class TextGridParser:
    def __init__(self, textgrid_lines, tier_name):
        self.tier_name = tier_name
        self.lastEnd = 0
        self.lines = textgrid_lines

    def parse(self):
        lines = [l.strip() for l in self.lines]

        size = None
        item = None

        i = 0
        intervals = {}

        while(not size):
            size = try_getting("size", lines[i])
            i += 1

        for item_number in range(0, int(size)):
            item = None
            while(not item):
                item_line = re.match(r"item \[(\d+)\]", lines[i])
                if item_line:
                    item = item_line.groups()[0]
                    i += 1
                    classs = get("class", lines[i])
                    i += 1
                    name = get("name", lines[i])
                    i += 1
                    # print item, classs, name
                    intervals[name] = []
                else:
                    i += 1

            if classs.startswith("TextTier"):
                continue

            assert_is("xmin", lines[i]); i+= 1
            assert_is("xmax", lines[i]); i+= 1
            n_intervals = int(get("intervals: size", lines[i])); i += 1

            for interval_id in range(0, n_intervals):
                assert_is("intervals", lines[i]); i+= 1
                xmin = get("xmin", lines[i]); i += 1
                xmax = get("xmax", lines[i]); i += 1
                text = get("text", lines[i]); i += 1
                intervals[name].append((float(xmin), float(xmax), text))

        def as_interval(y):
            return Interval(y[0], y[1], y[2])

        res = []
        for v in intervals[self.tier_name]:
            interval = as_interval(v)
            if interval.start != self.lastEnd or interval.end < self.lastEnd:
                raise ParseError("Missing time values")
            else:
                self.lastEnd = interval.end

            res.append(interval)

        return WaveSurfer(res)


def read(textgrid_file, tier_name):
    if not common.exists(textgrid_file):
        raise Exception("Missing file: " + textgrid_file)
    try:
        lines = codecs.open(textgrid_file, encoding='utf-8').readlines()
    except UnicodeDecodeError:
        print "textgrid file not UTF-8, converting"
        lib.system.run_command("iconv -f UTF-16 -t UTF-8 {} > tmp; mv tmp {}".format(textgrid_file, textgrid_file))
        lines = codecs.open(textgrid_file, encoding='utf-8').readlines()

    return TextGridParser(lines, tier_name).parse()
