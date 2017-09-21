# coding=utf-8


class FeatureExtractor(object):
    def extract(self, instance):
        raise NotImplementedError("{} must implement the 'extract' method".format(self))

    def __init__(self, config):
        raise NotImplementedError("{} must implement the '__init__' method".format(self))

    def batch_extract(self, instances):
        raise NotImplementedError("{} must implement the 'extract_batch' method".format(self))

    def feature_names(self):
        raise NotImplementedError("{} must implement the 'feature_names' method".format(self))
