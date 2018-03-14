import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser(description='Training.')

parser.add_argument('-ln', action='store_true', help='Use the use linear regression model')
parser.add_argument('-rid', action='store_true', help='Use the use linear regression model')
parser.add_argument('-poly', action='store_true', help='Use the use linear regression model')
parser.add_argument('-sgd', action='store_true', help='Use the use linear regression model')

args = parser.parse_args()

''' ################### Training ##################### '''
trainingX = np.loadtxt('x_train.txt')
trainingY = np.loadtxt('y_train.txt')

args.ln = True

if args.ln is True:
    print("Use linear regression model")
    reg_model = LinearRegression()

elif args.rid is True:
    print("Use Ridge regression model")
    fullPipeLine = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    trainingX = fullPipeLine.fit_transform(trainingX)
    reg_model = Ridge()

elif args.poly is True:  #
    print("Use Polynomial regression model")
    fullPipeLine = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    trainingX = fullPipeLine.fit_transform(trainingX)
    reg_model = LinearRegression()
elif args.sgd is True:
    print("Use Stochastic Gradient Descent model")
    fullPipeLine = Pipeline([
        ("stf_scale", StandardScaler())
    ])
    trainingX = fullPipeLine.fit_transform(trainingX)
    reg_model = SGDRegressor(penalty="l2")
else:
    print("Please select the regression model")
    quit()

reg_model.fit(trainingX, trainingY)

y_hat = reg_model.predict(trainingX)

# Save into file
joblib.dump(reg_model, "trainedReg.pkl")
scores = cross_val_score(reg_model, trainingX, trainingY, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Training result")
print("Scores", rmse)
print("RMSE:", rmse.mean())
print("Standard deviation", rmse.std())

# trainingX = np.c_[np.ones_like(trainingX),trainingX]
print('MSE = ', mean_squared_error(trainingY, y_hat))

''' ################ Testing ################## '''
testingX = np.loadtxt('x_test.txt')
testingY = np.loadtxt('y_test.txt')

# testingX = np.c_[np.ones_like(testingX),testingX]
if args.ln is False:
    testingX = fullPipeLine.fit_transform(testingX)

y_hat = reg_model.predict(testingX)

mse = mean_squared_error(testingY, y_hat)
rmse = np.sqrt(mse)
print("")
print("Testing result")
print("RMSE:", rmse.mean())
print('MSE = ', mse)
