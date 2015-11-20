from dataset import Dataset

"""
Data classes implement an interface for accessing feature values and class labels
of some dataset.

In this case, our dataset is represented as a list of lists, each of which
contain their features as the first N elements and their label as the last element.
"""
class ListDataset(Dataset):
    def value_of(self, attribute, point):
        return point[attribute]

    def outcome_of(self, point):
        return point[-1]

    def attributes(self):
        return range(len(self.points[0])-1)

if __name__ == '__main__':
    import unittest
    class TestListDataset(unittest.TestCase):
        def test_access(self):
            point1 = [1,2,1]
            point2 = [1,1,2]
            dataset = ListDataset([point1, point2])
            self.assertEqual(dataset.attributes(), [0, 1])
            self.assertEqual(dataset.outcome_of(point1), 1)
            self.assertEqual(dataset.outcome_of(point2), 2)
            self.assertEqual(dataset.value_of(0, point1), 1)
        def test_entropy(self):
            point1 = [0, 1, 'H']
            point2 = [0, 0, 'T']
            dataset = ListDataset([point1, point2])
            self.assertEqual(dataset.entropy_of(0), 1)
            self.assertEqual(dataset.entropy_of(1), 0)
            self.assertEqual(dataset.best_attribute(), 1)
            self.assertEqual(dataset.best_attribute([0]), 0)
            self.assertEqual(dataset.best_attribute([0, 1]), 1)
        def test_split_on(self):
            point1 = [0, 1, 'H']
            point2 = [0, 0, 'T']
            point3 = [1, 0, 'T']
            dataset = ListDataset([point1, point2, point3])
            split = dataset.split_on(1)
            self.assertEqual(split.keys(), [0, 1])
            self.assertEqual(split[1].points, [point1])
            self.assertEqual(split[0].points, [point2, point3])
        def test_most_common_outcome(self):
            point1 = [0, 1, 'H']
            point2 = [0, 0, 'T']
            point3 = [1, 0, 'T']
            dataset = ListDataset([point1, point2, point3])
            self.assertEqual(dataset.most_common_outcome(), 'T')

    unittest.main()
