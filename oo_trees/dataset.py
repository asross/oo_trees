from collections import defaultdict
from collections import Counter
from .outcome_counter import *
from .attribute import *
import random
import numpy

class Dataset():
    def __init__(self, X, y, attributes=None):
        self.X = X
        self.y = y
        self.outcome_counter = OutcomeCounter(y)
        self.attributes = attributes or [CategoricalAttribute(i) for i in range(X.shape[1])]
        assert self.X.shape[0] == len(y), "len(y) must match len(X)"
        assert self.X.shape[1] == len(self.attributes), "len(attributes) must match len(X[i])"

    def __len__(self):
        return self.X.shape[0]

    def entropy(self):
        return self.outcome_counter.entropy()

    def each_single_attribute_splitter(self):
        if not self.outcome_counter.is_unanimous():
            #for attribute in self.attributes:
            for attribute in random.sample(self.attributes, int(numpy.ceil(numpy.sqrt(len(self.attributes))))):
                for splitter in attribute.each_splitter(self.X[:, attribute.index]):
                    yield splitter

    def best_single_attribute_splitter(self):
        return self.best_splitter(self.each_single_attribute_splitter())

    def best_splitter(self, splitters):
        # in python 3+, this could just be `return min(splitters, key=self.splitter_entropy, default=None)`
        best_splitter = None
        min_entropy = float('inf')
        for splitter in splitters:
            entropy = self.splitter_entropy(splitter)
            if entropy < min_entropy:
                best_splitter = splitter
                min_entropy = entropy
        return best_splitter

    def splitter_entropy(self, splitter):
        splits = defaultdict(OutcomeCounter)
        for i in range(len(self)):
            splits[splitter.split(self.X[i])].record(self.y[i])
        return sum(y.total * y.entropy() for y in splits.values()) / float(len(self))

    def split_on(self, splitter):
        splits = defaultdict(list)
        for i in range(len(self)):
            splits[splitter.split(self.X[i])].append(i)
        return { value: self.take(indices) for value, indices in splits.items() }

    def take(self, indices):
        return self.__class__(self.X.take(indices, 0), self.y.take(indices), self.attributes)

    def random_split(self, fraction):
        n_examples = int(len(self) * fraction)
        indices = list(range(len(self)))
        random.shuffle(indices)
        return self.take(indices[:n_examples]), self.take(indices[n_examples:])

    def bootstrap(self, n=None):
        return self.take([random.randrange(len(self)) for _i in range(n or len(self))])
