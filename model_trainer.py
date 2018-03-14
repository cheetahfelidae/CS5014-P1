import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

training_x = np.loadtxt('x_train.txt')
training_y = np.loadtxt('y_train.txt')

training_x = np.c_[np.ones_like(training_x), training_x]
lin_reg = LinearRegression()
lin_reg.fit(training_x, training_y)
y_hat = lin_reg.predict(training_x)

# Save into file
joblib.dump(lin_reg, "trainedLinearReg.pkl")
scores = cross_val_score(lin_reg, training_x, training_y, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(training_y, y_hat))
