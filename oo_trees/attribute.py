import numpy
from .splitter import *

class Attribute():
    def __init__(self, index=None, name=''):
        if index is None: raise ValueError('must pass an index')
        self.index = index
        self.name = name

    def each_splitter(self, values):
        raise NotImplementedError

class CategoricalAttribute(Attribute):
    def each_splitter(self, values):
        unique_values = numpy.unique(values)
        if len(unique_values) > 1:
            for value in unique_values:
                yield IsEqualSplitter(self, value)

class NumericAttribute(Attribute):
    def each_splitter(self, values):
        stddev, maximum, value = values.std(), values.max(), values.min()
        while maximum > value:
            yield GreaterThanOrEqualToSplitter(self, value)
            value += stddev / 10.0 # TODO: this is arbitrary?
