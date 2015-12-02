from collections import defaultdict
from math import log

class EntropyCounter():
    def __init__(self):
        self.outcomes = defaultdict(int)
        self.total = 0

    def record(self, outcome):
        self.outcomes[outcome] += 1
        self.total += 1

    def entropy_of(self, outcome):
        probability = self.outcomes[outcome] / float(self.total)
        return -probability * log(probability, 2)

    def entropy(self):
        return sum(self.entropy_of(outcome) for outcome in self.outcomes.keys())

if __name__ == '__main__':
    import unittest

    class TestEntropyCounter(unittest.TestCase):
        def test_counting(self):
            counter = EntropyCounter()
            for c in 'aabbc': counter.record(c)
            self.assertEqual(counter.total, 5)
            self.assertEqual(counter.outcomes['a'], 2)
            self.assertEqual(counter.outcomes['b'], 2)
            self.assertEqual(counter.outcomes['c'], 1)

        def test_entropy(self):
            counter = EntropyCounter()
            self.assertEqual(counter.entropy(), 0)
            counter.record('H')
            self.assertEqual(counter.entropy(), 0)
            counter.record('T')
            self.assertEqual(counter.entropy(), 1)

    unittest.main()
