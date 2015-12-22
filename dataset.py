from collections import defaultdict
from collections import Counter
from entropy_counter import EntropyCounter
import random

class Dataset():
    def __init__(self, X, y=None):
       self.X = X
       self.y = y
       self.n_points = X.shape[0]

    def most_common_outcomes(self, n=None):
        return Counter(outcome for outcome in self.y).most_common(n)

    def most_common_outcome(self):
        return self.most_common_outcomes(1)[0][0]

    def is_unanimous(self):
        return len(self.most_common_outcomes()) == 1

    def take(self, indices):
        return self.__class__(self.X.take(indices, 0), self.y.take(indices))

    def features(self):
        return range(self.X.shape[1])

    def best_feature(self, features):
        return min(features, key=self.entropy_of)

    def entropy_of(self, feature):
        entropies = defaultdict(EntropyCounter)
        for i in range(self.n_points):
            entropies[self.X[i][feature]].record(self.y[i])
        return sum(counter.entropy() for counter in entropies.values())

    def split_on(self, feature):
        indices_by_value = defaultdict(list)
        for i in range(self.n_points):
            indices_by_value[self.X[i][feature]].append(i)
        return { value: self.take(indices) for value, indices in indices_by_value.items() }

    def bootstrap(self, n_points=None):
        indices = list()
        for _i in range(n_points or self.n_points):
            indices.append(random.randrange(self.n_points))
        return self.take(indices)

if __name__ == '__main__':
    import unittest
    import numpy as np

    class TestDataset(unittest.TestCase):
        def test_entropy(self):
            X = np.array([[0, 1], [0, 0]])
            y = np.array(['H', 'T'])
            dataset = Dataset(X, y)
            self.assertEqual(dataset.entropy_of(0), 1)
            self.assertEqual(dataset.entropy_of(1), 0)
            self.assertEqual(dataset.best_feature([0]), 0)
            self.assertEqual(dataset.best_feature([0, 1]), 1)

        def test_split_on(self):
            X = np.array([[0, 1], [0, 0], [1, 0]])
            y = np.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            split = dataset.split_on(1)
            self.assertEqual(split.keys(), [0, 1])
            np.testing.assert_array_equal(split[1].X, np.array([[0, 1]]))
            np.testing.assert_array_equal(split[0].X, np.array([[0, 0], [1, 0]]))

        def test_most_common_outcome(self):
            X = np.array([[0, 1], [0, 0], [1, 0]])
            y = np.array(['H', 'T', 'T'])
            dataset = Dataset(X, y)
            self.assertEqual(dataset.most_common_outcome(), 'T')

        def test_is_unanimous(self):
            unanimous_dataset = Dataset(np.array([[0, 0], [1, 0]]), np.array(['T', 'T']))
            fractious_dataset = Dataset(np.array([[0, 1], [0, 0], [1, 0]]), np.array(['H', 'T', 'T']))
            self.assertEqual(unanimous_dataset.is_unanimous(), True)
            self.assertEqual(fractious_dataset.is_unanimous(), False)

        def test_bootstrap(self):
            X = np.array([[0, 1], [0, 0]])
            y = np.array(['H', 'T'])
            dataset = Dataset(X, y)
            bootstrap = dataset.bootstrap(1000)
            self.assertEqual(bootstrap.n_points, 1000)
            self.assertEqual('H' in bootstrap.y, True) # this has a 10e-302ish chance of failing

    unittest.main()
