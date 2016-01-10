from collections import defaultdict
from collections import Counter
from outcome_counter import OutcomeCounter
from attribute import CategoricalAttribute, NumericAttribute
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

if __name__ == '__main__':
    import unittest
    from splitter import *

    class TestDataset(unittest.TestCase):
        def test_entropy(self):
            X = numpy.array([[0, 1], [0, 0]])
            y = numpy.array(['H', 'T'])
            dataset = Dataset(X, y)
            c0, c1 = dataset.attributes
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(c0, 0)), 1)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(c0, 1)), 1)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(c1, 0)), 0)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(c1, 1)), 0)

            best_splitter = dataset.best_single_attribute_splitter()
            self.assertEqual(best_splitter.attribute.index, 1)
            self.assertEqual(best_splitter.value, 0)

        def test_split_on(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            c0, c1 = dataset.attributes
            split = dataset.split_on(IsEqualSplitter(c1, 0))
            numpy.testing.assert_array_equal(split[0].X, numpy.array([[0, 1]]))
            numpy.testing.assert_array_equal(split[1].X, numpy.array([[0, 0], [1, 0]]))

        def test_multitype_splitting(self):
            # x1 < 0.5, x2 = 0 => 'Red'
            # x1 < 0.5, x2 = 1 => 'Yellow'
            # x1 >= .5 => 'Green'
            X = numpy.array([[0.25, 0], [0.33, 0], [0.31, 1], [0.12, 1], [0.45, 0], [0.52, 0], [0.81, 0], [0.67, 1], [0.51, 1]])
            y = numpy.array(['Red', 'Red', 'Yellow', 'Yellow', 'Red', 'Green', 'Green', 'Green', 'Green'])
            dataset = Dataset(X, y, [NumericAttribute(0), CategoricalAttribute(1)])
            splitter = dataset.best_single_attribute_splitter()
            self.assertEqual(splitter.attribute.index, 0)
            self.assertGreaterEqual(splitter.value, 0.45)
            self.assertLess(splitter.value, 0.52)

            subset1, subset2 = dataset.split_on(splitter).values()
            subsplitter = subset1.best_single_attribute_splitter()
            self.assertEqual(subsplitter.attribute.index, 1)
            self.assertEqual(subsplitter.value, 0)

        def test_more_complicated_splitting(self):
            # x1  < 0.25 => 'a'
            # x1 >= 0.25, x2 = 0 => 'b'
            # x1  < 0.50, x2 = 1 => 'c'
            # x1 >= 0.50, x2 = 1 => 'd'
            X = numpy.array([[0.2, 0], [0.01, 1], [0.15, 0], [0.232, 1], [0.173, 0], [0.263, 0], [0.671, 0], [0.9, 0], [0.387, 1], [0.482, 1], [0.632, 1], [0.892, 1]])
            y = numpy.array(['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'a', 'a'])
            dataset = Dataset(X, y, [NumericAttribute(0), CategoricalAttribute(1)])

            splitter = dataset.best_single_attribute_splitter()
            self.assertEqual(splitter.attribute.index, 0)
            self.assertGreaterEqual(splitter.value, 0.23)
            self.assertLess(splitter.value, 0.27)
            subset1, subset2 = dataset.split_on(splitter).values()
            numpy.testing.assert_array_equal(subset1.y, ['a', 'a', 'a', 'a', 'a'])

            splitter2 = subset2.best_single_attribute_splitter()
            self.assertEqual(splitter2.attribute.index, 1)
            self.assertEqual(splitter2.value, 0)
            subset21, subset22 = subset2.split_on(splitter2).values()
            numpy.testing.assert_array_equal(subset22.y, ['b', 'b', 'b'])

            splitter21 = subset21.best_single_attribute_splitter()
            self.assertEqual(splitter21.attribute.index, 0)
            self.assertGreaterEqual(splitter21.value, 0.482)
            self.assertLess(splitter21.value, 0.632)
            subset211, subset212 = subset21.split_on(splitter21).values()
            numpy.testing.assert_array_equal(subset211.y, ['c', 'c'])
            numpy.testing.assert_array_equal(subset212.y, ['a', 'a'])

        def test_outcomes(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            outcomes = dataset.outcome_counter
            self.assertEqual(outcomes.counter.most_common(), [('T', 2), ('H', 1)])

        def test_bootstrap(self):
            X = numpy.array([[0, 1], [0, 0]])
            y = numpy.array(['H', 'T'])
            dataset = Dataset(X, y)
            bootstrap = dataset.bootstrap(1000)
            self.assertEqual(bootstrap.X.shape[0], 1000)
            self.assertEqual('H' in bootstrap.y, True) # this has a 10e-302ish chance of failing

    unittest.main()
