import numpy as np
import sys
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

parser.add_argument('-lin', action='store_true', help='Linear Regression Model')
parser.add_argument('-rid', action='store_true', help='Ridge Regression Model')
parser.add_argument('-pol', action='store_true', help='Polynomial Regression Model')
parser.add_argument('-sto', action='store_true', help='Stochastic Gradient Descent Model')

args = parser.parse_args()

''' *********************************************** Training Part *********************************************** '''
training_x = np.loadtxt('training_x.txt')
training_y = np.loadtxt('training_y.txt')

if args.lin is True:
    print("Linear Regression Model")
    reg_model = LinearRegression()

elif args.rid is True:
    print("Ridge Regression Model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    reg_model = Ridge()

elif args.pol is True:
    print("Polynomial Regression Model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    reg_model = LinearRegression()
elif args.sto is True:
    print("Stochastic Gradient Descent Model")
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

# evaluate score by cross validation
scores = cross_val_score(reg_model, training_x, training_y, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Training result")
print("RMSE:", rmse)
print("RMSE (mean): %0.2f (+/- %0.2f)" % (rmse.mean(), rmse.std() * 2))
print("R^2: %0.2f" % (reg_model.score(training_x, training_y)))


''' *********************************************** Testing Part *********************************************** '''
testing_x = np.loadtxt('test_x.txt')
testing_y = np.loadtxt('test_y.txt')

if args.lin is False:
    testing_x = full_pipe_line.fit_transform(testing_x)

y_hat = reg_model.predict(testing_x)

rmse = np.sqrt(mean_squared_error(testing_y, y_hat))
print("")
print("Testing result")
print("RMSE: %0.2f" % (rmse.mean()))
print("R^2: %0.2f" % (reg_model.score(testing_x, testing_y)))
print("")
