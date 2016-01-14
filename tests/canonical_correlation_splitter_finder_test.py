import unittest
import numpy
from oo_trees.dataset import Dataset
from oo_trees.attribute import *
from oo_trees.canonical_correlation_splitter_finder import *

class TestCanonicalCorrelationSplitterFinder(unittest.TestCase):
    def test_linear_combination(self):
        attributes = [NumericAttribute(index=1), NumericAttribute(index=0)]
        weights = numpy.array([-1, 1])
        combination = LinearCombination(attributes, weights)
        self.assertEqual(combination.of(numpy.array([2, 1, 5])), 1)
        self.assertEqual(combination.of(numpy.array([1, 2, 7])), -1)

    def test_cca(self):
        X = numpy.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = numpy.array([0, 0, 1, 1])
        dataset = Dataset(X, y)

        splitter_finder = CanonicalCorrelationSplitterFinder(dataset, n=2)
        numpy.testing.assert_array_almost_equal([numpy.sqrt(2)/2, numpy.sqrt(2)/2], splitter_finder.linear_combination.weights)

        best_splitter = splitter_finder.best_splitter()

        self.assertGreater(best_splitter.value, numpy.sqrt(2) - 0.01)
        self.assertLess(best_splitter.value, numpy.sqrt(2) + 0.01)
