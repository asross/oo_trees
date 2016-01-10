from collections import Counter
from .classifier import *
from .axis_aligned_decision_tree import *

class RandomForest(Classifier):
    def __init__(self, dataset, tree_class=AxisAlignedDecisionTree, n_trees=10, examples_per_tree=None):
        self.trees = [tree_class(dataset.bootstrap(examples_per_tree)) for _i in range(n_trees)]

    def vote_on(self, x):
        # TODO: we could return early as soon as we have a definite plurality
        return Counter(tree.classify(x) for tree in self.trees)

    def classify(self, x):
        return self.vote_on(x).most_common(1)[0][0]
