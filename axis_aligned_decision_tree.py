class AxisAlignedDecisionTree:
    def __init__(self, dataset, attributes=None):
        # the root node looks at every attribute
        if attributes is None: attributes = dataset.attributes()

        if len(attributes) is 0 or dataset.is_unanimous():
            self.leaf_outcome = dataset.most_common_outcome()
        else:
            self.leaf_outcome = None
            self.branch_attribute = dataset.best_attribute(attributes)
            remaining_attributes = list(attributes)
            remaining_attributes.remove(self.branch_attribute)
            self.branches = { branch_value: self.__class__(subset, remaining_attributes) for branch_value, subset in dataset.split_on(self.branch_attribute).items() }

    def classify(self, point):
        if self.leaf_outcome:
            return self.leaf_outcome
        else:
            branch_value = point.get(self.branch_attribute)
            branch_tree = self.branches[branch_value]
            return branch_tree.classify(point)

if __name__ == '__main__':
    import unittest
    from list_datapoint import ListDatapoint
    from dataset import Dataset
    class TestAxisAlignedDecisionTree(unittest.TestCase):
        def test_classification(self):
            point1 = ListDatapoint([0, 1, 'H'])
            point2 = ListDatapoint([0, 0, 'H'])
            point3 = ListDatapoint([1, 0, 'H'])
            point4 = ListDatapoint([1, 1, 'T'])
            dataset = Dataset([point1, point2, point3, point4])
            tree = AxisAlignedDecisionTree(dataset)
            self.assertEqual(tree.classify(ListDatapoint([0, 0])), 'H')
            self.assertEqual(tree.classify(ListDatapoint([0, 1])), 'H')
            self.assertEqual(tree.classify(ListDatapoint([1, 0])), 'H')
            self.assertEqual(tree.classify(ListDatapoint([1, 1])), 'T')
    unittest.main()
