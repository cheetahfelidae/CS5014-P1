import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

training_x = np.loadtxt('training_x.txt')
training_y = np.loadtxt('training_y.txt')

training_x = np.c_[np.ones_like(training_x), training_x]

# fit a model on the training data and trying to predict the test data
lin_reg = LinearRegression()
lin_reg.fit(training_x, training_y)
y_hat = lin_reg.predict(training_x)

joblib.dump(lin_reg, "trained_linear_reg.pkl")

# evaluate score by cross validation
scores = cross_val_score(lin_reg, training_x, training_y, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Accuracy:", rmse)
print("Accuracy (mean): %0.2f (+/- %0.2f)" % (rmse.mean(), rmse.std() * 2))
print("MSE:", mean_squared_error(training_y, y_hat))
