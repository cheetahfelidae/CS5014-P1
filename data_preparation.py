from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def cal_NSM(date):
    tomorrow = date + timedelta(1)
    midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - date).seconds


# load the data
energy = pd.read_csv('energydata_complete.csv')

# convert the column of datetime to NSM format which will be used for the prediction.
NSMs = []
for index, row in energy.iterrows():
    NSMs.append(cal_NSM(datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")))
energy["NSM"] = NSMs

# ignore some features (independent variables) with statistically low correlation with the energy consumption of appliances (dependent variable)
energy.drop(["rv1", "rv2", "date"], axis=1, inplace=True)

energy.hist(bins=50, figsize=(20, 15))
plt.show()

print("Correlation: ")
print(energy.corr()['Appliances'].sort_values(ascending=False))
print("")

print("Statistical Descriptions: ")
print(energy.describe())

# split data into random train and test subsets
# the proportion of the train to test data is 80 to 20
training_set, testing_set = train_test_split(energy, test_size=0.2, random_state=42)

# train data set
training_x = training_set.drop("Appliances", axis=1)
training_y = training_set["Appliances"].copy()

np.savetxt("training_x.txt", training_x)
np.savetxt("training_y.txt", training_y)

# test data set
test_x = testing_set.drop("Appliances", axis=1)
test_y = testing_set["Appliances"].copy()

np.savetxt("test_x.txt", test_x)
np.savetxt("test_y.txt", test_y)
