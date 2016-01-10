import unittest
import numpy
from oo_trees.dataset import Dataset
from oo_trees.attribute import *
from oo_trees.splitter import *

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
