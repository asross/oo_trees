from collections import Counter
from classifier import Classifier
from axis_aligned_decision_tree import AxisAlignedDecisionTree

class RandomForest(Classifier):
    def __init__(self, dataset, tree_class=AxisAlignedDecisionTree, n_trees=10, examples_per_tree=None):
        self.trees = [tree_class(dataset.bootstrap(examples_per_tree)) for _i in range(n_trees)]

    def vote_on(self, x):
        # TODO: we could return early as soon as we have a definite plurality
        return Counter(tree.classify(x) for tree in self.trees)

    def classify(self, x):
        return self.vote_on(x).most_common(1)[0][0]

if __name__ == '__main__':
    import unittest
    from dataset import Dataset
    import numpy

    class TestRandomForest(unittest.TestCase):
        def test_classification(self):
            X = numpy.array([[0], [0], [1], [1], [1]])
            y = numpy.array(['H', 'H', 'H', 'T', 'T'])
            dataset = Dataset(X, y)
            forest = RandomForest(dataset, n_trees=500, examples_per_tree=5)
            self.assertEqual(forest.classify([0]), 'H')
            self.assertEqual(forest.classify([1]), 'T')
            forest = RandomForest(dataset, n_trees=5, examples_per_tree=500)
            self.assertEqual(forest.classify([0]), 'H')
            self.assertEqual(forest.classify([1]), 'T')

    unittest.main()
