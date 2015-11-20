from collections import defaultdict
from math import log

class EntropyCounter():
    def __init__(self):
        self.dictionary = defaultdict(int)
        self.total = 0
    def record(self, value):
        self.dictionary[value] += 1
        self.total += 1
    def entropy_of(self, count):
        probability = float(count) / self.total
        return -probability * log(probability, 2)
    def entropy(self):
        return sum(self.entropy_of(count) for _, count in self.dictionary.items())

if __name__ == '__main__':
    import unittest
    class TestEntropyCounter(unittest.TestCase):
        def test_counting(self):
            counter = EntropyCounter()
            for c in 'aabbc': counter.record(c)
            self.assertEqual(counter.total, 5)
            self.assertEqual(counter.dictionary['a'], 2)
            self.assertEqual(counter.dictionary['b'], 2)
            self.assertEqual(counter.dictionary['c'], 1)
        def test_entropy(self):
            counter = EntropyCounter()
            self.assertEqual(counter.entropy(), 0)
            counter.record('H')
            self.assertEqual(counter.entropy(), 0)
            counter.record('T')
            self.assertEqual(counter.entropy(), 1)
    unittest.main()
