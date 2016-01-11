from collections import defaultdict
from collections import Counter
from .outcome_counter import *
from .attribute import *
from .single_attribute_splitter_finder import *
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

    def best_single_attribute_splitter(self):
        finder = SingleAttributeSplitterFinder(self, n=len(self.attributes))
        return finder.best_splitter()

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
