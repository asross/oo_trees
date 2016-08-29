import numpy
import random
from .splitter import *
from sklearn.cross_decomposition import CCA

class LinearCombination():
    def __init__(self, attributes, weights):
        self.attributes = attributes
        self.weights = weights

    def of(self, x):
        return self.weights.dot(x.take([a.index for a in self.attributes]))

class CanonicalCorrelationSplitterFinder():
    def __init__(self, dataset, n=None, tol=1e-4):
        if n is None:
            n = int(numpy.ceil(numpy.sqrt(len(dataset.attributes))))

        self.dataset = dataset
        self.attributes = random.sample(dataset.attributes, n)

        cca = CCA(n_components=1, tol=tol)
        cca.fit(
            dataset.X.take([a.index for a in self.attributes], 1),
            dataset.y)

        self.linear_combination = LinearCombination(
            self.attributes,
            cca.x_weights_.transpose()[0])

    def each_splitter(self):
        values = numpy.array([self.linear_combination.of(x) for x in self.dataset.X])
        stddev, maximum, value = values.std(), values.max(), values.min()
        while maximum > value:
            yield LinearCombinationGreaterThanOrEqualToSplitter(self.linear_combination, value)
            value += stddev / 10.0 # TODO: this is arbitrary?

    def best_splitter(self):
        return min(self.each_splitter(), key=self.dataset.splitter_entropy,
                   default=None)
