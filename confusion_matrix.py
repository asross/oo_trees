class ConfusionMatrix:
    #            Real 
    #        y1  y2  y3  y4
    #Pred y1  5   1   0   2
    #
    #     y2  2  10   3   1
    #
    #     y3  8   1   5   0
    #
    #     y4  0   0   0   5

    def confusion_about(self, outcome):
        return None

    def precision(self):
        return None

    def __init__(self, predicted_y, real_y):
        # self.n_predictions = 0
        # self.n_correct = 0

        # for y, i in enumerate(real_y):
            # correct = y == predicted_y[i]
            # self.n_predictions += 1
            # self.n_correct += correct


