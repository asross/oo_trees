class AxisAlignedDecisionTree:
    def __init__(self, dataset, features=None):
        # the root node looks at every feature
        if features is None: features = dataset.features()

        # record the most common outcome
        self.most_common_outcome = dataset.most_common_outcome()

        # potentially add some branches
        self.branches = {}
        if len(features) >= 1 and not dataset.is_unanimous():
            best_feature = dataset.best_feature(features)
            remaining_features = [a for a in features if a != best_feature]
            self.feature = best_feature
            for value, subset in dataset.split_on(best_feature).items():
                self.branches[value] = self.__class__(subset, remaining_features)

    def classify(self, point):
        if self.branches:
            value = point[self.feature]
            if value in self.branches:
                # we have a subtree for the point's feature value
                return self.branches[value].classify(point)
            else:
                # we have subtrees, but none for the point; we could try to pick
                # a subtree but for now we'll just return the most popular leaf.
                return self.most_common_outcome
        else:
            # we don't have any subtrees (i.e. we're a leaf node)
            return self.most_common_outcome

if __name__ == '__main__':
    import unittest
    import numpy as np
    from dataset import Dataset

    class TestAxisAlignedDecisionTree(unittest.TestCase):
        def test_classification(self):
            X = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
            y = np.array(['H', 'H', 'H', 'T'])
            dataset = Dataset(X, y)
            tree = AxisAlignedDecisionTree(dataset)
            self.assertEqual(tree.classify([0, 0]), 'H')
            self.assertEqual(tree.classify([0, 1]), 'H')
            self.assertEqual(tree.classify([1, 0]), 'H')
            self.assertEqual(tree.classify([1, 1]), 'T')
            self.assertEqual(tree.classify([2, 0]), 'H') # it can handle unknown values too

    unittest.main()
