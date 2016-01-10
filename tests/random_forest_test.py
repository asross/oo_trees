import unittest
import numpy
from oo_trees.dataset import *
from oo_trees.random_forest import *

class TestRandomForest(unittest.TestCase):
    def test_classification(self):
        X = numpy.array([[0], [0], [1], [1], [1]])
        y = numpy.array(['H', 'H', 'H', 'T', 'T'])
        dataset = Dataset(X, y)
        forest = RandomForest(dataset, n_trees=500, examples_per_tree=5)
        self.assertEqual(forest.classify([0]), 'H')
        self.assertEqual(forest.classify([1]), 'T')
        forest = RandomForest(dataset, n_trees=5, examples_per_tree=500)
        self.assertEqual(forest.classify([0]), 'H')
        self.assertEqual(forest.classify([1]), 'T')
