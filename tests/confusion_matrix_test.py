import unittest
import numpy
from oo_trees.confusion_matrix import *

class TestConfusionMatrix(unittest.TestCase):
    def test_accuracy(self):
        cm = ConfusionMatrix([0, 0, 1, 1],
                             [1, 0, 1, 0])
        numpy.testing.assert_almost_equal(cm.accuracy, 0.5)

    def test_specificity(self):
        cm = ConfusionMatrix([0, 0, 2, 1, 1, 0],
                             [0, 1, 2, 1, 2, 1])
        numpy.testing.assert_almost_equal(cm.specificity_for(2), 1)
        numpy.testing.assert_almost_equal(cm.specificity_for(1), 2/3.0)
        numpy.testing.assert_almost_equal(cm.specificity_for(0), 3/5.0)
        numpy.testing.assert_almost_equal(cm.specificity(), 34/45.0)

    def test_sensitivity(self):
        cm = ConfusionMatrix([0, 0, 2, 1, 1, 0],
                             [0, 1, 2, 1, 2, 1])
        numpy.testing.assert_almost_equal(cm.sensitivity_for(2), 1/2.0)
        numpy.testing.assert_almost_equal(cm.sensitivity_for(1), 1/3.0)
        numpy.testing.assert_almost_equal(cm.sensitivity_for(0), 1)
        numpy.testing.assert_almost_equal(cm.sensitivity(), 11/18.0)

    def test_more_complicated_case(self):
        pred = numpy.array(['a', 'a', 'c', 'c', 'a', 'b', 'b', 'a', 'c', 'c'])
        real = numpy.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])
        cm = ConfusionMatrix(pred, real)
        numpy.testing.assert_almost_equal(cm.sensitivity_for('a'), 1/2.0)
        numpy.testing.assert_almost_equal(cm.sensitivity_for('b'), 2/3.0)
        numpy.testing.assert_almost_equal(cm.sensitivity_for('c'), 2/3.0)
        numpy.testing.assert_almost_equal(cm.specificity_for('a'), 4/6.0)
        numpy.testing.assert_almost_equal(cm.specificity_for('b'), 1)
        numpy.testing.assert_almost_equal(cm.specificity_for('c'), 5/7.0)

    def test_to_array(self):
        cm = ConfusionMatrix([0, 0, 2, 2, 0, 2],
                             [2, 0, 2, 2, 0, 1])
        numpy.testing.assert_array_equal(cm.to_array(), [[2, 0, 0],
                                                         [0, 0, 1],
                                                         [1, 0, 2]])
