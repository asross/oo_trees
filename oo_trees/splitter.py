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
