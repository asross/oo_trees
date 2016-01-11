# Decision Trees

This repository will contain several variants of decision tree / ensemble
classification algorithms, written in an object-oriented style. My immediate
goal is to try to reproduce some of the results from
[this paper](http://arxiv.org/abs/1507.05444) on canonical correlation forests.

Where possible, external parameters names will match scikit-learn's
implementations of
[decision trees](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
and [random forests](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Usage

```python
X = examples
y = outcomes
attributes = [
  CategoricalAttribute(0, 'color'),
  NumericAttribute(1, 'latitude'),
  NumericAttribute(2, 'longitude')
]

dataset = oo_trees.Dataset(X, y)
