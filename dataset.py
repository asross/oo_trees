from collections import defaultdict
from collections import Counter
from entropy_counter import EntropyCounter
import random
import numpy
import operator

class Dataset():
    def __init__(self, X, y=None, attribute_types=None, all_categorical=False):
       self.X = X
       self.y = y
       if attribute_types is None:
          self.attribute_types = numpy.full(self.X.shape[1], all_categorical)
       else:
          self.attribute_types = attribute_types

    def is_numeric(self, attribute):
        return self.attribute_types[attribute]

    def each_splitter(self):
        for attribute in range(self.X.shape[1]):
            values = self.X[:, attribute]
            if self.is_numeric(attribute):
                stdd, maximum, value = values.std(), values.max(), values.min()
                while value < maximum:
                    yield attribute, operator.gt, value
                    value += stdd / 10.0
            else:
                for value in numpy.unique(values):
                    yield attribute, operator.eq, value

    def best_splitter(self):
        return min(self.each_splitter(), key=self.entropy_of)

    def entropy_of(self, splitter):
        attribute, fn, value = splitter
        split_entropies = [EntropyCounter(), EntropyCounter()]
        for i in range(self.X.shape[0]):
            split_entropies[fn(self.X[i][attribute], value)].record(self.y[i])
        return sum(split.entropy() for split in split_entropies)

    def split_on(self, splitter):
        attribute, fn, value = splitter
        splits = [[], []]
        for i in range(self.X.shape[0]):
            splits[fn(self.X[i][attribute], value)].append(i)
        return map(lambda i: self.take(i), splits)

    def most_common_outcomes(self, n=None):
        return Counter(outcome for outcome in self.y).most_common(n)

    def most_common_outcome(self):
        return self.most_common_outcomes(1)[0][0]

    def is_unanimous(self):
        return len(self.most_common_outcomes()) == 1

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
            dataset = Dataset(X, y)
            self.assertEqual(dataset.entropy_of([0, operator.eq, 0]), 1)
            self.assertEqual(dataset.entropy_of([0, operator.eq, 1]), 1)
            self.assertEqual(dataset.entropy_of([1, operator.eq, 0]), 0)
            self.assertEqual(dataset.entropy_of([1, operator.eq, 1]), 0)
            self.assertEqual(dataset.best_splitter(), (1, operator.eq, 0))

        def test_split_on(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            split = dataset.split_on([1, operator.eq, 0])
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
            splitter = dataset.best_splitter()
            attr, op, value = splitter
            self.assertEqual(attr, 0)
            self.assertEqual(op, operator.gt)
            self.assertGreaterEqual(value, 0.45)
            self.assertLess(value, 0.52)

            subset1, subset2 = dataset.split_on(splitter)
            subsplitter = subset1.best_splitter()
            sattr, sop, svalue = subsplitter
            self.assertEqual(sattr, 1)
            self.assertEqual(sop, operator.eq)
            self.assertEqual(svalue, 0)

        def test_most_common_outcome(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0]])
            y = numpy.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            self.assertEqual(dataset.most_common_outcome(), 'T')

        def test_is_unanimous(self):
            unanimous_dataset = Dataset(numpy.array([[0, 0], [1, 0]]), numpy.array(['T', 'T']))
            fractious_dataset = Dataset(numpy.array([[0, 1], [0, 0], [1, 0]]), numpy.array(['H', 'T', 'T']))
            self.assertEqual(unanimous_dataset.is_unanimous(), True)
            self.assertEqual(fractious_dataset.is_unanimous(), False)

        def test_bootstrap(self):
            X = numpy.array([[0, 1], [0, 0]])
            y = numpy.array(['H', 'T'])
            dataset = Dataset(X, y)
            bootstrap = dataset.bootstrap(1000)
            self.assertEqual(bootstrap.X.shape[0], 1000)
            self.assertEqual('H' in bootstrap.y, True) # this has a 10e-302ish chance of failing

    unittest.main()
