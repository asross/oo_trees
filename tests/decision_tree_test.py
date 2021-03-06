import unittest
import numpy
from oo_trees.dataset import *
from oo_trees.decision_tree import *
from oo_trees.attribute import *

class TestDecisionTree(unittest.TestCase):
    def test_classification(self):
        X = numpy.array([[0, 1], [0, 0], [1, 0], [1, 1]])
        y = numpy.array(['H', 'H', 'H', 'T'])
        dataset = Dataset(X, y)
        tree = DecisionTree(dataset)
        self.assertEqual(len(tree.branches), 2)
        self.assertEqual(len(tree.branches[1].branches), 0)
        self.assertEqual(len(tree.branches[0].branches), 2)
        self.assertEqual(len(tree.branches[0].branches[1].branches), 0)
        self.assertEqual(len(tree.branches[0].branches[0].branches), 0)

        self.assertEqual(tree.classify([0, 0]), 'H')
        self.assertEqual(tree.classify([0, 1]), 'H')
        self.assertEqual(tree.classify([1, 0]), 'H')
        self.assertEqual(tree.classify([1, 1]), 'T')
        self.assertEqual(tree.classify([2, 0]), 'H') # it can handle unknown values too

    def test_min_points(self):
        X = numpy.array([[0], [1], [1]])
        y = numpy.array(['H', 'T', 'T'])
        dataset = Dataset(X, y)
        tree = DecisionTree(dataset, min_samples_split=0)
        self.assertEqual(len(tree.branches), 2)
        tree = DecisionTree(dataset, min_samples_split=5)
        self.assertEqual(len(tree.branches), 0)
        self.assertEqual(tree.leaf_value(), 'T')

    def test_max_depth(self):
        X = numpy.array([[0], [1], [1]])
        y = numpy.array(['H', 'T', 'T'])
        dataset = Dataset(X, y)
        tree = DecisionTree(dataset, max_depth=3)
        self.assertEqual(len(tree.branches), 2)
        numpy.testing.assert_array_equal([2, 2],
                [t.depth for t in tree.branches.values()])
        tree = DecisionTree(dataset, max_depth=1)
        self.assertEqual(len(tree.branches), 0)
        self.assertEqual(tree.leaf_value(), 'T')

    def test_performance_on(self):
        # x1  < 0.25 => 'a'
        # x1 >= 0.25, x2 = 0 => 'b'
        # x1  < 0.50, x2 = 1 => 'c'
        # x1 >= 0.50, x2 = 1 => 'a'
        Xtrain = numpy.array([[0.15, 0], [0.232, 1], [0.173, 0], [0.263, 0], [0.671, 0], [0.9, 0], [0.387, 1], [0.482, 1], [0.632, 1], [0.892, 1]])
        ytrain = numpy.array([      'a',        'a',        'a',        'b',        'b',      'b',        'c',        'c',        'a',        'a'])
        training_dataset = Dataset(Xtrain, ytrain, [NumericAttribute(0), CategoricalAttribute(1)])
        tree = DecisionTree(training_dataset)

        # expecting
        #         Real
        #       a   b   c
        #Pred a 2   0   2
        #
        #     b 1   2   0
        #
        #     c 1   0   2
        # accuracy:  6/10
                              # a,a       a,a         a,c       a,c         b,a      b,b        b,b       c,a       c,c         c,c
        Xtest = numpy.array([[0.13, 0], [0.73, 1], [0.47, 1], [0.33, 1], [0.7, 1], [0.3, 0], [0.5, 0], [0.1, 1], [0.476, 1], [0.265, 1]])
        ytest = numpy.array([      'a',       'a',       'a',       'a',      'b',      'b',      'b',      'c',        'c',        'c'])
        test_dataset = Dataset(Xtest, ytest, [NumericAttribute(0), CategoricalAttribute(1)])

        performance = tree.performance_on(test_dataset)
        self.assertEqual(performance.accuracy, 0.6)
        numpy.testing.assert_array_equal(performance.to_array(), [[2,0,2], [1,2,0], [1,0,2]])
