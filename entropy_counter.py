from collections import defaultdict
from math import log

class EntropyCounter():
    def __init__(self):
        self.outcome_counts = defaultdict(int)
        self.total = 0
    def record(self, outcome):
        self.outcome_counts[outcome] += 1
        self.total += 1
    def outcome_entropy(self, outcome):
        probability = self.outcome_counts[outcome] / float(self.total)
        return -probability * log(probability, 2)
    def entropy(self):
        return sum(self.outcome_entropy(outcome) for outcome in self.outcome_counts.keys())

if __name__ == '__main__':
    import unittest
    class TestEntropyCounter(unittest.TestCase):
        def test_counting(self):
            counter = EntropyCounter()
            for c in 'aabbc': counter.record(c)
            self.assertEqual(counter.total, 5)
            self.assertEqual(counter.outcome_counts['a'], 2)
            self.assertEqual(counter.outcome_counts['b'], 2)
            self.assertEqual(counter.outcome_counts['c'], 1)
        def test_entropy(self):
            counter = EntropyCounter()
            self.assertEqual(counter.entropy(), 0)
            counter.record('H')
            self.assertEqual(counter.entropy(), 0)
            counter.record('T')
            self.assertEqual(counter.entropy(), 1)
    unittest.main()
