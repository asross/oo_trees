# Object-Oriented Decision Trees

This repository will contain several variants of decision tree / ensemble
classification algorithms, written in an object-oriented style. My immediate
goal is to try to reproduce some of the results from
[this paper](http://arxiv.org/abs/1507.05444) on canonical correlation forests,
which I am testing against
[the same datasets](https://bitbucket.org/twgr/ccf/src/master/Datasets).

Where possible, external parameters names will match `scikit-learn`'s
implementations of
[decision trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
and [random forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Usage

One major difference from `scikit-learn` is that datasets and their attributes
are treated as first-class objects. Additionally, all classifiers must be
initialized with their training dataset (as opposed to calling `fit`).

```python
from oo_trees.dataset import Dataset
from oo_trees.decision_tree import DecisionTree
from oo_trees.random_forest import RandomForest

X = examples # numpy 2D numeric array
y = outcomes # numpy 1D array

dataset = Dataset(X, y)

training_dataset, test_dataset = dataset.random_split(0.75)

d_tree = DecisionTree(training_dataset)
forest = RandomForest(training_dataset)

print(d_tree.classify(test_dataset.X[0]))
print(forest.classify(test_dataset.X[0]))

d_tree_confusion_matrix = d_tree.performance_on(test_dataset)
forest_confusion_matrix = forest.performance_on(test_dataset)

print(d_tree_confusion_matrix.accuracy)
print(forest_confusion_matrix.accuracy)
```

For canonical correlation trees, you can dependency-inject the behavior:

```python
from oo_trees.canonical_correlation_splitter_finder import *

cc_tree = DecisionTree(training_dataset,
    splitter_finder=CanonicalCorrelationSplitterFinder)
```

This interface will hopefully become more elegant soon.

When initializing datasets, we assume all attributes of the training examples
are categorical. If that is not the case, you can pass in an additional
`attribute_types` variable on initialize:

```python
from oo_trees.dataset import Dataset
from oo_trees.attribute import NumericAttribute, CategoricalAttribute

X = examples
y = outcomes

attributes = [
  NumericAttribute(index=0, name='age'),
  CategoricalAttribute(index=1, name='sex'),
  NumericAttribute(index=2, name='income')
]

dataset = Dataset(X, y, attributes)
```

The logic for finding the best split is differs for each attribute type, and in
the future there may be additional type-specific parameters (such as
importance or number-to-name mappings) useful for classification or display.
