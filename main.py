import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def display_scores(scores):
    print("Scores", scores)
    print("Mean", scores.mean())
    print("Standard deviation", scores.std())


def get_scores(model, x_train, y_train, scoring_val="neg_mean_squared_error", cv_val=10):
    scores = cross_val_score(model, x_train, y_train, scoring=scoring_val, cv=cv_val)
    return scores

x_train = np.loadtxt("x_train")
x_train = np.c_[np.ones_like(x_train), x_train]
y_train = np.loadtxt("y_train")

linerg = LinearRegression()
linerg.fit(x_train, y_train)
joblib.dump(linerg, "normal_linreg.pkl")
scores = get_scores(linerg, x_train, y_train)
rmse_scores = np.sqrt(-scores)
display_scores(rmse_scores)