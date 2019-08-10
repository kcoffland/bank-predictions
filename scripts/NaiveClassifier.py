from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class NaiveClassifier:
    """
    A Naive classifier which finds the majority class, and only predicts that
    """

    def __init__(self):
        """Constructor, sets that values for the current instance
        """
        self.is_fit = False
        self.majority_val = 0

    def fit(self, X, y=None):
        """Finds which class is the majority class in the training data
        Args:
            X: DataFrame, Doesn't matter in the slightest for this model
            y: Series, the target column, must be given for the model to 
                        be used, but is equal to none initially for the paradigm
        Returns:
            self: NaiveModel instance, for the fit->transform paradigm
        """
        if type(y) == None or type(y) != pd.core.series.Series:
            raise TypeError(f"y must be given as a series, not {type(y)}")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        unique, counts = np.unique(y_enc, return_counts=True)
        del unique
        self.majority_val = np.argmax(counts)
        
        self.is_fit = True

        return self

    def transform(self, X, y=None):
        """Does nothing since the data doesn't need to be cleaned at all
        Args:
            X: dataframe, holds all feature data
            y: series, target data
        Returns:
            X_new: dataframe, transformed feature data
        """

        if not self.is_fit:
            raise RuntimeError("Data must be fit before it can be transformed")
        X_new = X
        return X_new

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def predict(self, X):
        """Takes in data and returns an array filled of the majority class for each row
        Args:
            X: DataFrame, Data to predict on
        Returns:
            y_pred: ndarray, an array of size corresponding to the number of 
                    elements that need to be predicted on. It is full of the 
                    majority class.
        """
        shape = X.shape[0]
        y_pred = np.full(shape, self.majority_val)
        return y_pred