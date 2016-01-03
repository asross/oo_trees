from confusion_matrix import ConfusionMatrix

class Classifier():
    def __init__(self, training_dataset):
        raise NotImplementedError

    def classify(self, point):
        raise NotImplementedError

    def performance_on(self, test_dataset):
        predictions = map(self.classify, test_dataset.X)
        realities = test_dataset.y
        return ConfusionMatrix(predictions, realities)
