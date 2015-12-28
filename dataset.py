from collections import defaultdict
from collections import Counter
from outcome_counter import OutcomeCounter
import random
import numpy

class SingleAttributeSplitter():
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

class GreaterThanSplitter(SingleAttributeSplitter):
    def split(self, point):
        return point[self.attribute] > self.value

class IsEqualSplitter(SingleAttributeSplitter):
    def split(self, point):
        return point[self.attribute] == self.value

class Dataset():
    def __init__(self, X, y=None, attribute_types=None):
        self.X = X
        self.y = y
        self.attribute_types = attribute_types
        if attribute_types is None:
            self.attribute_types = numpy.full(self.X.shape[1], 0)

    def is_numeric(self, attribute):
        return self.attribute_types[attribute]

    def each_single_attribute_splitter(self):
        for attribute in range(self.X.shape[1]):
            values = self.X[:, attribute]
            if self.is_numeric(attribute):
                stddev, maximum, value = values.std(), values.max(), values.min()
                while value < maximum:
                    yield GreaterThanSplitter(attribute, value)
                    value += stddev / 10.0
            else:
                for value in numpy.unique(values):
                    yield IsEqualSplitter(attribute, value)

    def best_single_attribute_splitter(self):
        return min(self.each_single_attribute_splitter(), key=self.splitter_entropy)

    def splitter_entropy(self, splitter):
        splits = [OutcomeCounter(), OutcomeCounter()]
        for i in range(self.X.shape[0]):
            splits[splitter.split(self.X[i])].record(self.y[i])
        return sum(split.entropy() for split in splits)

    def split_on(self, splitter):
        splits = [[], []]
        for i in range(self.X.shape[0]):
            splits[splitter.split(self.X[i])].append(i)
        return [self.take(split) for split in splits]

    def outcomes(self):
        return Counter(self.y)

    def take(self, indices):
        return self.__class__(self.X.take(indices, 0), self.y.take(indices), self.attribute_types)

    def bootstrap(self, n_points=None):
        indices = []
        for _i in range(n_points or self.X.shape[0]):
            indices.append(random.randrange(self.X.shape[0]))
        return self.take(indices)

if __name__ == '__main__':
    import unittest

    class TestDataset(unittest.TestCase):
        def test_entropy(self):
            X = numpy.array([[0, 1], [0, 0]])
            y = numpy.array(['H', 'T'])
            dataset = Dataset(X, y, [0, 0])
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(0, 0)), 1)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(0, 1)), 1)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(1, 0)), 0)
            self.assertEqual(dataset.splitter_entropy(IsEqualSplitter(1, 1)), 0)

            best_splitter = dataset.best_single_attribute_splitter()
            self.assertEqual(best_splitter.attribute, 1)
            self.assertEqual(best_splitter.value, 0)

        def test_split_on(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            split = dataset.split_on(IsEqualSplitter(1, 0))
            numpy.testing.assert_array_equal(split[0].X, numpy.array([[0, 1]]))
            numpy.testing.assert_array_equal(split[1].X, numpy.array([[0, 0], [1, 0]]))

        def test_multitype_splitting(self):
            # x1 < 0.5, x2 = 0 => 'Red'
            # x1 < 0.5, x2 = 1 => 'Yellow'
            # x1 >= .5 => 'Green'
            X = numpy.array([[0.25, 0],
                             [0.33, 0],
                             [0.31, 1],
                             [0.12, 1],
                             [0.45, 0],
                             [0.52, 0],
                             [0.81, 0],
                             [0.67, 1],
                             [0.51, 1]])
            y = numpy.array(['Red', 'Red', 'Yellow', 'Yellow', 'Red', 'Green', 'Green', 'Green', 'Green'])
            dataset = Dataset(X, y, [1, 0])
            splitter = dataset.best_single_attribute_splitter()
            self.assertEqual(splitter.attribute, 0)
            self.assertGreaterEqual(splitter.value, 0.45)
            self.assertLess(splitter.value, 0.52)

            subset1, subset2 = dataset.split_on(splitter)
            subsplitter = subset1.best_single_attribute_splitter()
            self.assertEqual(subsplitter.attribute, 1)
            self.assertEqual(subsplitter.value, 0)

        def test_outcomes(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            outcomes = dataset.outcomes()
            self.assertEqual(outcomes.most_common(), [('T', 2), ('H', 1)])

        def test_bootstrap(self):
            X = numpy.array([[0, 1], [0, 0]])
            y = numpy.array(['H', 'T'])
            dataset = Dataset(X, y)
            bootstrap = dataset.bootstrap(1000)
            self.assertEqual(bootstrap.X.shape[0], 1000)
            self.assertEqual('H' in bootstrap.y, True) # this has a 10e-302ish chance of failing

    unittest.main()
