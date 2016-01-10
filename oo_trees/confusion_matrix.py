# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy
from .outcome_counter import *

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
        return numpy.array([[self.counts[o][oo] for oo in self.outcomes] for o in self.outcomes])

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
        return numpy.mean([self.sensitivity_for(o) for o in self.outcomes])

    def specificity(self):
        return numpy.mean([self.specificity_for(o) for o in self.outcomes])
