from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self, dataset, min_samples_split=2):
        self.min_samples_split = min_samples_split
        self.outcome_counter = dataset.outcome_counter
        self.branches = {}
        self.splitter = None
        if len(dataset) >= min_samples_split:
            self.grow_branches(dataset)

    def grow_branches(self, dataset):
        raise NotImplementedError

    def branch_for(self, x):
        # we are a leaf node unless we have a splitter
        if self.splitter:
            value = self.splitter.split(x)
            # if we have a missing value (impossible for binary splits, but
            # possible for other subclasses), return most common value
            if self.branches.has_key(value):
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
