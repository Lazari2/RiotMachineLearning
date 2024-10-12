import numpy as np
from Accuracy import Accuracy

class Accuracy_Binary(Accuracy):

    def init(self, y):
        pass

    def compare(self, predictions, y):
        predictions = np.round(predictions)
        return predictions == y