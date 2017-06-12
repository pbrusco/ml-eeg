# coding=utf-8

import ml.system as system
import re

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
    def __init__(self, textgrid_lines):
        self.lines = [l.strip() for l in textgrid_lines]

    def next(self):
        if self.pointer >= len(self.lines):
            raise Exception("parser reach end unexpectedly")
        else:
            self.pointer += 1

    def current(self):
        return self.lines[self.pointer]

    def get_and_move(self):
        v = self.current()
        self.next()
        return v

    def find_item(self):
        item = None
        while(not item):
            item_line = re.match(r"item \[(\d+)\]", self.current())

            if item_line:
                item = item_line.groups()[0]
                self.next()
                classs = get("class", self.get_and_move())
                name = get("name", self.get_and_move())
                # print item, classs, name
                self.tuples[name] = []
                return classs, name
            else:
                self.next()

    def parse_interval_tier(self):
        res = []
        assert_is("xmin", self.get_and_move())
        assert_is("xmax", self.get_and_move())
        n_intervals = int(get("intervals: size", self.get_and_move()))

        for interval_id in range(0, n_intervals):
            assert_is("intervals", self.get_and_move())
            xmin = get("xmin", self.get_and_move())
            xmax = get("xmax", self.get_and_move())
            text = get("text", self.get_and_move())
            res.append((float(xmin), float(xmax), text))

        return res

    def parse_point_tier(self):
        res = []
        assert_is("xmin", self.get_and_move())
        assert_is("xmax", self.get_and_move())
        n_intervals = int(get("points: size", self.get_and_move()))

        for interval_id in range(0, n_intervals):
            assert_is("points", self.get_and_move())
            number = get("number", self.get_and_move())
            mark = get("mark", self.get_and_move())
            res.append((float(number), mark))
        return res

    def parse(self):
        size = None

        self.pointer = 0
        self.tuples = {}

        while(not size):
            size = try_getting("size", self.get_and_move())

        for item_number in range(0, int(size)):
            classs, name = self.find_item()

            if classs.startswith("TextTier"):
                self.tuples[name] = self.parse_point_tier()

            elif classs.startswith("IntervalTier"):
                self.tuples[name] = self.parse_interval_tier()

        return dict([(tier, self.tuples[tier]) for tier in self.tuples.keys()])


def read(textgrid_file):
    if not system.exists(textgrid_file):
        raise Exception("Missing file: " + textgrid_file)
    try:
        lines = codecs.open(textgrid_file, encoding='utf-8').readlines()
    except UnicodeDecodeError:
        print("textgrid file not UTF-8, converting")
        system.run_command("iconv -f UTF-16 -t UTF-8 {} > tmp; mv tmp {}".format(textgrid_file, textgrid_file))
        lines = codecs.open(textgrid_file, encoding='utf-8').readlines()

    return TextGridParser(lines).parse()
