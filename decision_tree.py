from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self, dataset, min_samples_split=2, max_depth=float('inf'), depth=1):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.outcome_counter = dataset.outcome_counter
        self.branches = {}
        self.splitter = None
        if max_depth > depth and len(dataset) >= min_samples_split:
            self.grow_branches(dataset)

    def new_branch(self, dataset):
        return self.__class__(dataset,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                depth=self.depth+1)

    def grow_branches(self, dataset):
        raise NotImplementedError

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
