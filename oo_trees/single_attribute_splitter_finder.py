import random
import numpy

class SingleAttributeSplitterFinder():
    def __init__(self, dataset, n=None):
        self.dataset = dataset
        if n is None:
            n = int(numpy.ceil(numpy.sqrt(len(dataset.attributes))))
        self.attributes = random.sample(dataset.attributes, n)

    def each_splitter(self):
        for attribute in self.attributes:
            attr_values = self.dataset.X[:, attribute.index]
            for splitter in attribute.each_splitter(attr_values):
                yield splitter

    def best_splitter(self):
        return min(self.each_splitter(), key=self.dataset.splitter_entropy,
                   default=None)
