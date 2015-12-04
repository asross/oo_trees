from collections import Counter
from axis_aligned_decision_tree import AxisAlignedDecisionTree
import multiprocessing as mp

def grow_tree(tree_class, dataset, n_points):
    return tree_class(dataset.bootstrap(n_points))

class DecisionForest:
    def __init__(self, dataset, tree_class=None, n_trees=None, n_points=None, processor=None, processes=None):
        if n_trees is None: n_trees = 50 # TODO -- how many?
        if n_points is None: n_points = len(dataset.points)
        if processes is None: processes = 10 # TODO -- how many?
        if processor is None: processor = mp.Pool(processes=processes)
        if tree_class is None: tree_class = AxisAlignedDecisionTree

        self.trees = [processor.apply(grow_tree, (tree_class, dataset, n_points)) for _i in range(n_trees)]

    def vote_on(self, point):
        # TODO: we could return early as soon as we have a definite plurality
        return Counter(tree.classify(point) for tree in self.trees)

    def classify(self, point):
        return self.vote_on(point).most_common(1)[0][0]

if __name__ == '__main__':
    import unittest
    from dataset import Dataset
    from list_datapoint import ListDatapoint

    class TestDecisionForest(unittest.TestCase):
        def test_classification(self):
            point1 = ListDatapoint([0, 'H'])
            point2 = ListDatapoint([0, 'H'])
            point3 = ListDatapoint([1, 'H'])
            point4 = ListDatapoint([1, 'T'])
            point5 = ListDatapoint([1, 'T'])
            dataset = Dataset([point1, point2, point3, point4, point5])
            forest = DecisionForest(dataset, n_trees=500, n_points=5)
            self.assertEqual(forest.classify(point1), 'H')
            self.assertEqual(forest.classify(point5), 'T')
            self.assertEqual(forest.classify(point3), 'T')
            forest = DecisionForest(dataset, n_trees=5, n_points=500)
            self.assertEqual(forest.classify(point1), 'H')
            self.assertEqual(forest.classify(point5), 'T')
            self.assertEqual(forest.classify(point3), 'T')

    unittest.main()
