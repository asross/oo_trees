class ListDatapoint():
    def __init__(self, datalist):
        self.datalist = datalist
    def get(self, attribute):
        return self.datalist[attribute]
    def outcome(self):
        return self.datalist[-1]
    def attributes(self):
        return range(len(self.datalist)-1)
