import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

trainingX = np.loadtxt("x_train.txt")
trainingY = np.loadtxt("y_train.txt")

fullPipeLine = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)), ("stf_scale", StandardScaler())
])

trainingX = fullPipeLine.fit_transform(trainingX)
linreg = LinearRegression()
linreg.fit(trainingX, trainingY)

y_hat = linreg.predict(trainingX)
# Save into file
joblib.dump(linreg, "trainedLinearReg.pkl")
scores = cross_val_score(linreg, trainingX, trainingY, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(trainingY, y_hat))

''' ################### Testing ################## '''
testingX = np.loadtxt('x_test.txt')
testingY = np.loadtxt('y_test.txt')

testingX = fullPipeLine.fit_transform(testingX)

y_hat = linreg.predict(testingX)

scores = cross_val_score(linreg, testingX, testingY, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)
print("")
print("Testing result")
print("Scores", rmse)
print("Mean:", rmse.mean())
print("Standard deviation", rmse.std())
print('MSE = ', mean_squared_error(testingY, y_hat))
