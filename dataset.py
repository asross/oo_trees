from collections import defaultdict
from entropy_counter import EntropyCounter

class Dataset():
    def __init__(self, points=[]):
        self.points = points

    def entropy_of(self, attribute):
        entropies = defaultdict(EntropyCounter)
        for point in self.points:
            entropies[self.value_of(attribute, point)].record(self.outcome_of(point))
        return sum(counter.entropy() for counter in entropies.values())

    def best_attribute(self, attributes=None):
        if attributes is None: attributes = self.attributes()
        return min(attributes, key=self.entropy_of)

    def split_on(self, attribute):
        points_by_value = defaultdict(list)
        for point in self.points:
            points_by_value[self.value_of(attribute, point)].append(point)
        return { value: self.__class__(points) for value, points in points_by_value.items() }

    def value_of(self, attribute, point):
        raise NotImplementedError

    def outcome_of(self, point):
        raise NotImplementedError

    def attributes(self):
        raise NotImplementedError
