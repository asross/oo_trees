from .classifier import *
from .single_attribute_splitter_finder import *

class DecisionTree(Classifier):
    def __init__(self, dataset,
                 min_samples_split=2,
                 max_depth=float('inf'),
                 depth=1,
                 splitter_finder=SingleAttributeSplitterFinder):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.outcome_counter = dataset.outcome_counter
        self.splitter = None
        self.splitter_finder = splitter_finder
        self.branches = {}

        # Stop recursing if we're at a leaf
        if depth >= max_depth: return
        if len(dataset) < min_samples_split: return
        if len(self.outcome_counter) == 1: return

        # Otherwise, branch
        splitter = splitter_finder(dataset).best_splitter()
        if splitter:
            self.splitter = splitter
            self.branches = { value: self.new_branch(subset)
                for value, subset in dataset.split_on(splitter).items() }

    def new_branch(self, dataset):
        return self.__class__(dataset,
                splitter_finder=self.splitter_finder,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                depth=self.depth+1)

    def branch_for(self, x):
        # we are a leaf node unless we have a splitter
        if self.splitter:
            value = self.splitter.split(x)
            # if we have a missing value (impossible for binary splits, but
            # possible for other subclasses), return most common value
            if value in self.branches:
                return self.branches[value]
        return None

    def leaf_value(self):
        return self.outcome_counter.most_common_value()

    def classify(self, x):
        branch = self.branch_for(x)
        if branch:
            return branch.classify(x)
        else:
            return self.leaf_value()
