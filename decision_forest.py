from collections import Counter

class DecisionForest:
    def __init__(self, dataset, tree_class, tree_count, point_count=None):
        self.trees = [tree_class(dataset.bootstrap(point_count)) for i in range(tree_count)]

    def classify(self, point):
        outcomes = Counter(tree.classify(point) for tree in self.trees)
        return outcomes.most_common(1)[0][0]

if __name__ == '__main__':
    import unittest
    from dataset import Dataset
    from list_datapoint import ListDatapoint
    from axis_aligned_decision_tree import AxisAlignedDecisionTree

    class TestDecisionForest(unittest.TestCase):
        def test_classification(self):
            point1 = ListDatapoint([0, 'H'])
            point2 = ListDatapoint([0, 'H'])
            point3 = ListDatapoint([1, 'H'])
            point4 = ListDatapoint([1, 'T'])
            point5 = ListDatapoint([1, 'T'])
            dataset = Dataset([point1, point2, point3, point4, point5])
            forest = DecisionForest(dataset, AxisAlignedDecisionTree, 500, 5)
            self.assertEqual(forest.classify(point1), 'H')
            self.assertEqual(forest.classify(point5), 'T')
            self.assertEqual(forest.classify(point3), 'T')
            forest = DecisionForest(dataset, AxisAlignedDecisionTree, 5, 500)
            self.assertEqual(forest.classify(point1), 'H')
            self.assertEqual(forest.classify(point5), 'T')
            self.assertEqual(forest.classify(point3), 'T')

    unittest.main()
