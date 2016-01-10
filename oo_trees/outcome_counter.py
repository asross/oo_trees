from collections import Counter
from math import log

class OutcomeCounter():
    def __init__(self, outcomes=[]):
        self.counter = Counter()
        self.total = 0
        for outcome in outcomes:
            self.record(outcome)

    def record(self, outcome):
        self.counter[outcome] += 1
        self.total += 1

    def entropy_of(self, outcome):
        probability = self[outcome] / float(self.total)
        return -probability * log(probability, 2)

    def outcomes(self):
        return self.counter.keys()

    def entropy(self):
        return sum(map(self.entropy_of, self.outcomes()))

    def is_unanimous(self):
        return len(self.counter) == 1

    def most_common_value(self):
        return self.counter.most_common(1)[0][0]

    def __getitem__(self, outcome):
        return self.counter[outcome]
