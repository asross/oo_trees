from .decision_tree import *

class AxisAlignedDecisionTree(DecisionTree):
    def grow_branches(self, dataset):
        # split until we have unanimity in either X or y
        self.splitter = dataset.best_single_attribute_splitter()
        if self.splitter:
            self.branches = { value: self.new_branch(subset)
                for value, subset in dataset.split_on(self.splitter).items() }
