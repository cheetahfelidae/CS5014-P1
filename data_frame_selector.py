class DataFrameSelector():
    def __init__(self, attributeNames):
        self.attributeNames = attributeNames

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributeNames].values
