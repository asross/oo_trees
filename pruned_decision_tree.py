from classifier import Classifier
from axis_aligned_decision_tree import AxisAlignedDecisionTree

class PrunedDecisionTree(Classifier):
    def __init__(self, dataset, tree_class=AxisAlignedDecisionTree, split_fraction=0.75):
        training_set, pruning_set = dataset.random_split(split_fraction)
        leafy_tree = tree_class(training_set)
        self.tree = self.greedily_prune(leafy_tree, pruning_set)

    def classify(self, x):
        return self.tree.classify(x)

    def greedily_prune(self, tree, dataset):
        self.initial_performance = tree.performance_on(dataset)
        self.best_performance = self.initial_performance
        for branch in tree.depth_order_traversal():
            splitter = branch.splitter
            if splitter:
                branch.splitter = None
                performance = tree.performance_on(dataset)
                if performance.accuracy > self.best_performance.accuracy:
                    self.best_performance = performance
                    branch.branches = {}
                else:
                    branch.splitter = splitter
        return tree
