class Splitter():
    def split(self, x):
        raise NotImplementedError

class SingleAttributeSplitter(Splitter):
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value

class IsEqualSplitter(SingleAttributeSplitter):
    def split(self, x):
        return x[self.attribute.index] == self.value

class GreaterThanOrEqualToSplitter(SingleAttributeSplitter):
    def split(self, x):
        return x[self.attribute.index] >= self.value

class LinearCombinationGreaterThanOrEqualToSplitter(Splitter):
    def __init__(self, linear_combination, value):
        self.linear_combination = linear_combination
        self.value = value

    def split(self, x):
        return self.linear_combination.of(x) >= self.value
