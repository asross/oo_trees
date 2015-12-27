class AxisAlignedDecisionTree:
    def __init__(self, dataset):
        outcomes = dataset.outcomes()
        self.most_common_outcome = outcomes.most_common(1)[0][0]

        # split until we have unanimity in either X or y
        self.splitter = None
        self.branches = []
        if len(outcomes) > 1:
            splitter = dataset.best_single_attribute_splitter()
            subset1, subset2 = dataset.split_on(splitter)
            if len(subset1.y) and len(subset2.y):
                self.splitter = splitter
                self.branches = [self.__class__(subset1), self.__class__(subset2)]

    def classify(self, point):
        if self.splitter:
            return self.branches[self.splitter.split(point)].classify(point)
        else:
            return self.most_common_outcome

if __name__ == '__main__':
    import unittest
    import numpy
    from dataset import Dataset

    class TestAxisAlignedDecisionTree(unittest.TestCase):
        def test_classification(self):
            X = numpy.array([[0, 1], [0, 0], [1, 0], [1, 1]])
            y = numpy.array(['H', 'H', 'H', 'T'])
            dataset = Dataset(X, y, [0, 0])
            tree = AxisAlignedDecisionTree(dataset)
            self.assertEqual(len(tree.branches), 2)
            self.assertEqual(len(tree.branches[1].branches), 0)
            self.assertEqual(len(tree.branches[0].branches), 2)
            self.assertEqual(len(tree.branches[0].branches[1].branches), 0)
            self.assertEqual(len(tree.branches[0].branches[0].branches), 0)

            self.assertEqual(tree.classify([0, 0]), 'H')
            self.assertEqual(tree.classify([0, 1]), 'H')
            self.assertEqual(tree.classify([1, 0]), 'H')
            self.assertEqual(tree.classify([1, 1]), 'T')
            self.assertEqual(tree.classify([2, 0]), 'H') # it can handle unknown values too

    unittest.main()
