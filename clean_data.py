import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def calculate_nms(dt_obj):
    tomorrow = dt_obj + timedelta(1)
    midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - dt_obj).seconds


energy = pd.read_csv('/Users/cheetah/Sites/CS5014-P1/energydata_complete.csv')
nsm_list = []
energy["nsm"] = ""
for index, row in energy.iterrows():
    dt_str = row["date"]
    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    nsm_list.append(calculate_nms(dt_obj))

energy["nsm"] = nsm_list
energy.drop(["rv1", "rv2", "date", "Visibility", "RH_5"], axis=1, inplace=True)

energy.hist(bins=50, figsize=(20, 15))
# energy.info()
# plt.show()
print(energy.corr()['Appliances'].sort_values(ascending=False))
print(energy.describe())

train_set, test_set = train_test_split(energy, test_size=0.2, random_state=42)
x_train = train_set.drop("Appliances", axis=1)
y_train = train_set["Appliances"].copy()

xTest = test_set.drop("Appliances", axis=1)
yTest = test_set["Appliances"].copy()

np.savetxt("x_train.txt", x_train)
np.savetxt("y_train.txt", y_train)

np.savetxt("x_test.txt", xTest)
np.savetxt("y_test.txt", yTest)
