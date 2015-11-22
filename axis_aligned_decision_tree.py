class AxisAlignedDecisionTree:
    def __init__(self, dataset, attributes=None):
        # the root node looks at every attribute
        if attributes is None: attributes = dataset.attributes()

        # record the most common outcome
        self.most_common_outcome = dataset.most_common_outcome()

        # potentially add some branches
        self.branches = {}
        if len(attributes) >= 1 and not dataset.is_unanimous():
            self.branch_attribute = dataset.best_attribute(attributes)
            remaining_attributes = [a for a in attributes if a != self.branch_attribute]
            for branch_value, subset in dataset.split_on(self.branch_attribute).items():
                self.branches[branch_value] = self.__class__(subset, remaining_attributes)

    def classify(self, point):
        if self.branches:
            branch_value = point.get(self.branch_attribute)
            if branch_value in self.branches:
                # we have a subtree for the point's branch attr value
                return self.branches[branch_value].classify(point)
            else:
                # we have subtrees, but none for the point; we could try to pick
                # a subtree but for now we'll just return the most popular leaf.
                return self.most_common_outcome
        else:
            # we don't have any subtrees (i.e. we're a leaf node)
            return self.most_common_outcome

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
            self.assertEqual(tree.classify(ListDatapoint([2, 0])), 'H') # it can handle unknown values too
    unittest.main()
