import unittest
from oo_trees.outcome_counter import *

class TestOutcomeCounter(unittest.TestCase):
    def test_counting(self):
        counter = OutcomeCounter()
        for c in 'aabbc': counter.record(c)
        self.assertEqual(counter.total, 5)
        self.assertEqual(counter['a'], 2)
        self.assertEqual(counter['b'], 2)
        self.assertEqual(counter['c'], 1)

    def test_entropy(self):
        counter = OutcomeCounter()
        self.assertEqual(counter.entropy(), 0)
        counter.record('H')
        self.assertEqual(counter.entropy(), 0)
        counter.record('T')
        self.assertEqual(counter.entropy(), 1)
