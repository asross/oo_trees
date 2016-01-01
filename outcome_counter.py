from collections import Counter
from math import log

class OutcomeCounter():
    def __init__(self, outcomes=[]):
        self.counter = Counter()
        self.total = 0
        for outcome in outcomes:
            self.record(outcome)

    def record(self, outcome):
        self.counter[outcome] += 1
        self.total += 1

    def entropy_of(self, outcome):
        probability = self[outcome] / float(self.total)
        return -probability * log(probability, 2)

    def outcomes(self):
        return self.counter.keys()

    def entropy(self):
        return sum(map(self.entropy_of, self.outcomes()))

    def weighted_entropy(self):
        return self.total * self.entropy()

    def is_unanimous(self):
        return len(self.counter) == 1

    def most_common_value(self):
        return self.counter.most_common(1)[0][0]

    def __getitem__(self, outcome):
        return self.counter[outcome]

    def __len__(self):
        return self.total

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
