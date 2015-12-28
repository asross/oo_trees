from collections import Counter
from math import log

class OutcomeCounter():
    def __init__(self):
        self.counter = Counter()
        self.total = 0

    def record(self, outcome):
        self.counter[outcome] += 1
        self.total += 1

    def entropy_of(self, outcome):
        probability = self[outcome] / float(self.total)
        return -probability * log(probability, 2)

    def outcomes(self):
        return self.counter.keys()

    def entropy(self):
        return sum(self.entropy_of(outcome) for outcome in self.outcomes())

    def __getitem__(self, outcome):
        return self.counter[outcome]

if __name__ == '__main__':
    import unittest

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

    unittest.main()
