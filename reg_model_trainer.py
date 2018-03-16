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
training_x = np.loadtxt('training_x.txt')
training_y = np.loadtxt('training_y.txt')

if args.ln is True:
    print("Use linear regression model")
    reg_model = LinearRegression()

elif args.rid is True:
    print("Use Ridge regression model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    reg_model = Ridge()

elif args.poly is True:
    print("Use Polynomial regression model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    reg_model = LinearRegression()
elif args.sgd is True:
    print("Use Stochastic Gradient Descent model")
    full_pipe_line = Pipeline([
        ("stf_scale", StandardScaler())
    ])
    training_x = full_pipe_line.fit_transform(training_x)
    reg_model = SGDRegressor(penalty="l2")
else:
    print("Please select the regression model")
    quit()

reg_model.fit(training_x, training_y)

y_hat = reg_model.predict(training_x)

# Save into file
joblib.dump(reg_model, "trained_reg.pkl")
scores = cross_val_score(reg_model, training_x, training_y, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Training result")
print("Scores", rmse)
print("RMSE:", rmse.mean())
print("Standard deviation", rmse.std())

# trainingX = np.c_[np.ones_like(trainingX),trainingX]
print('MSE = ', mean_squared_error(training_y, y_hat))

''' ################ Testing ################## '''
testing_x = np.loadtxt('test_x.txt')
testing_y = np.loadtxt('test_y.txt')

# testingX = np.c_[np.ones_like(testingX),testingX]
if args.ln is False:
    testing_x = full_pipe_line.fit_transform(testing_x)

y_hat = reg_model.predict(testing_x)

mse = mean_squared_error(testing_y, y_hat)
rmse = np.sqrt(mse)
print("")
print("Testing result")
print("RMSE:", rmse.mean())
print('MSE = ', mse)
