# Decision Trees

## Datasets

Datasets have a 2D numpy array of examples `X` and optionally a 1D numpy array of outcomes `y`.

Each example is a vector with
For each attribute in each example x∈X, we can ask

Dataset X, y, attributes

Attribute name, index, type

SingleAttributeSplitter attribute, value

Classifier
  - RandomForest
  - DecisionTree splitter, branches
    - AxisAlignedDecisionTree
    - CanonicalCorrelationTree



IDEAS:

## Probability Distribution Tree

What if you had a decision tree where, instead of returning a class, you return a probability distribution over classes, and secondly, splitting was non-deterministic and you took a weighted sum?

E.g. at each node splitter returns a value `p` between 0 and 1, and at each leaf node you return a distribution of outcomes instead of just one. Then you combine them like:

```
p * true_branch.outcome_distribution + (1-p) * false_branch.outcome_distribution
```

Perhaps the value of `p` should in some way be related to the quantity of information gained from the split. If you have a very informative split, p should be very close to 1, but if you learn little it should be closer to 0.5

## CCA only for numerical attributes

Is there any really good reason for converting categorical attributes to numerical ones using their indexes? Why not only do CCA with linear combinations of the numerical attributes, maybe in concert with particular settings of categorical attributes, as the split criteria?
