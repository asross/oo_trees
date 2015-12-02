class ListDatapoint():
    def __init__(self, data):
        self.data = data

    def get(self, feature):
        return self.data[feature]

    def outcome(self):
        return self.data[-1]

    def features(self):
        return range(len(self.data)-1)
