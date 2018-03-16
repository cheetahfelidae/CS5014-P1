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

parser.add_argument('-lin', action='store_true', help='Linear Regression Model')
parser.add_argument('-lri', action='store_true', help='Linear Ridge Regression Model')
parser.add_argument('-pri', action='store_true', help='Polynomial Ridge Regression Model')
parser.add_argument('-pol', action='store_true', help='Polynomial Regression Model')
parser.add_argument('-sto', action='store_true', help='Stochastic Gradient Descent Model')

args = parser.parse_args()

''' *********************************************** Training Part *********************************************** '''
training_x = np.loadtxt('training_x.txt')
training_y = np.loadtxt('training_y.txt')

if args.lin is True:
    print("Linear Regression Model")
    model = LinearRegression()

elif args.lri is True:
    print("Linear Ridge Regression Model")
    full_pipe_line = Pipeline([
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    model = Ridge()

elif args.pri is True:
    print("Polynomial Ridge Regression Model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    model = Ridge()

elif args.pol is True:
    print("Polynomial Regression Model")
    full_pipe_line = Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("stf_scale", StandardScaler())
    ])

    training_x = full_pipe_line.fit_transform(training_x)
    model = LinearRegression()
elif args.sto is True:
    print("Stochastic Gradient Descent Model")
    full_pipe_line = Pipeline([
        ("stf_scale", StandardScaler())
    ])
    training_x = full_pipe_line.fit_transform(training_x)
    model = SGDRegressor(penalty="l2")
else:
    print("Please enter the desired regression model")
    quit()

model.fit(training_x, training_y)

y_hat = model.predict(training_x)

# Save into file
joblib.dump(model, "trained_reg.pkl")

# evaluate score by cross validation
scores = cross_val_score(model, training_x, training_y, scoring="neg_mean_squared_error", cv=10)
rmse = np.sqrt(-scores)

print("Training result")
print("RMSE:", rmse)
print("RMSE (mean): %0.2f (+/- %0.2f)" % (rmse.mean(), rmse.std() * 2))
print("R^2: %0.2f" % (model.score(training_x, training_y)))


''' *********************************************** Testing Part *********************************************** '''
testing_x = np.loadtxt('test_x.txt')
testing_y = np.loadtxt('test_y.txt')

if args.lin is False:
    testing_x = full_pipe_line.fit_transform(testing_x)

y_hat = model.predict(testing_x)

rmse = np.sqrt(mean_squared_error(testing_y, y_hat))
print("")
print("Testing result")
print("RMSE: %0.2f" % (rmse.mean()))
print("R^2: %0.2f" % (model.score(testing_x, testing_y)))
print("")
