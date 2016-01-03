# -*- coding: utf-8 -*-

from collections import defaultdict
from outcome_counter import OutcomeCounter
import numpy

class ConfusionMatrix:
    #            Real
    #        y1  y2  y3  y4
    #Pred y1  5   1   0   2
    #
    #     y2  2  10   3   1
    #
    #     y3  8   1   5   0
    #
    #     y4  0   0   0   5
    def __init__(self, predictions, realities):
        self.counts = defaultdict(OutcomeCounter)
        outcomes = set()
        n_correct = 0
        self.total = 0
        for i in range(len(realities)):
            yr = realities[i]
            yp = predictions[i]
            outcomes |= set([yr, yp])
            n_correct += yr == yp
            self.counts[yr].record(yp)
            self.total += 1
        self.accuracy = n_correct / float(self.total)
        self.outcomes = sorted(list(outcomes))

    def to_array(self):
        return numpy.array(map(lambda o: map(lambda oo: self.counts[o][oo],
          self.outcomes), self.outcomes))

    def true_positives(self, outcome):
        return self.counts[outcome][outcome]

    def true_negatives(self, outcome):
        return sum(self.counts[o].total - self.counts[o][outcome]
            for o in self.outcomes if o != outcome)

    def false_positives(self, outcome):
        return sum(self.counts[o][outcome]
            for o in self.outcomes if o != outcome)

    def false_negatives(self, outcome):
        return self.counts[outcome].total - self.counts[outcome][outcome]

    def sensitivity_for(self, outcome):
        # probability of predicting outcome given real value is outcome
        # high sensitivity =>
        #   classifier rules out outcome if we didn't predict it
        return self.true_positives(outcome) / float(self.counts[outcome].total)

    def specificity_for(self, outcome):
        # probability of predicting ¬outcome given real value is ¬outcome
        # high specificity =>
        #   classifier rules in outcome if we did predict it
        return self.true_negatives(outcome) / float(self.total - self.counts[outcome].total)

    def sensitivity(self):
        return numpy.mean(map(self.sensitivity_for, self.outcomes))

    def specificity(self):
        return numpy.mean(map(self.specificity_for, self.outcomes))

if __name__ == '__main__':
    import unittest

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

    unittest.main()
