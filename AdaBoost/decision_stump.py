import numpy as np
import pandas as pd

class Decision_Stump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X: pd.DataFrame):
        # Set initial result
        feature_stump = X.iloc[:, self.feature_index]
        pred = np.ones(X.shape[0])

        if self.polarity == 1:
            pred[feature_stump < self.threshold] = -1
        else:
            pred[feature_stump > self.threshold] = -1

        return pred