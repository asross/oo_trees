from collections import Counter
from classifier import Classifier
from axis_aligned_decision_tree import AxisAlignedDecisionTree
import multiprocessing

def grow_tree(tree_class, dataset, n):
    return tree_class(dataset.bootstrap(n))

class DecisionForest(Classifier):
    def __init__(self, dataset, tree_class=AxisAlignedDecisionTree, n_trees=200, n_points=None, n_processes=1):
        if n_processes == 1:
            self.trees = [tree_class(dataset.bootstrap(n_points)) for _i in range(n_trees)]
        else:
            pool = multiprocessing.Pool(processes=n_processes)
            self.trees = [pool.apply(grow_tree, (tree_class, dataset, n_points)) for _i in range(n_trees)]

    def vote_on(self, x):
        # TODO: we could return early as soon as we have a definite plurality
        return Counter(tree.classify(x) for tree in self.trees)

    def classify(self, x):
        return self.vote_on(x).most_common(1)[0][0]

if __name__ == '__main__':
    import unittest
    from dataset import Dataset
    import numpy as np

    class TestDecisionForest(unittest.TestCase):
        def test_classification(self):
            X = np.array([[0], [0], [1], [1], [1]])
            y = np.array(['H', 'H', 'H', 'T', 'T'])
            dataset = Dataset(X, y)
            forest = DecisionForest(dataset, n_trees=500, n_points=5)
            self.assertEqual(forest.classify([0]), 'H')
            self.assertEqual(forest.classify([1]), 'T')
            forest = DecisionForest(dataset, n_trees=5, n_points=500)
            self.assertEqual(forest.classify([0]), 'H')
            self.assertEqual(forest.classify([1]), 'T')

    unittest.main()
