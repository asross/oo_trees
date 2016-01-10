## Questions

* What is the ideal split % of a full dataset into training and test?
* Should the training-test split be completely random or random but constrained?

My first impulse is to say that both the training and test sets should have at
least a few examples for each possible outcome, but if there are multiple rules
that could generate those possible outcomes, just having at least some number of
examples per outcome might not be sufficient.

Let's say we have some underlying set of rules, `underlying_rule_set`, that,
possibly with noise, actually generated the dataset. We could implement such a
generator pretty easily, and we could measure the extent to which a decision
tree trained on the dataset outputted by that generator faithfully encodes its
set of rules.

One question we might ask here is what quantity of randomly-selected data (for
a given `underlying_rule_set`) would we need to have, e.g., a >99% chance that
the dataset contains, with some minimum redundancy, examples for each rule.

Although if the rules are themselves redundant / not fully independent, maybe
we should require a lower quantity.

In either case, there is some question we need to ask about the complexity of
the underlying process that generated the data before we can say confidently
how much data we need to train a tree or any other classifier.

This is in danger of becoming circular. I want to ask if we can make any
statements about the complexity of that underlying process (or message source)
from the messages it outputs -- which we might quantify as the minimum
computational resources (space and time) necessary to store and train a good
classifier (a kind of Kolmogorov complexity). But training a classifier is the
whole point.

If we had software that could somehow do this kind of analysis, what might the
method signatures look like?

```python
class RandomRuleSet(IIDMessageSource):
    def __init__(self, attributes=[], n_rules=10, noise=0.01):

class IIDMessageSource(MessageSource):

class MessageSource():
    def entropy(self): # how much information do we gain from one message?
    def random_example(self):
    def random_dataset(self, n_examples=100):
    def min_examples_to_ensure(self, min_redundancy=2, with_likelihood=0.99)

rule_set = RandomRuleSet(attributes=attributes, n_rules=20, noise=0.05)
print rule_set.min_examples_to_ensure(min_redundancy=3, with_likelihood=0.95)
```
