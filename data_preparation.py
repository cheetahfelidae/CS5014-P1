from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

energy = pd.read_csv('energydata_complete.csv')


def cal_num_sec_to_mid(date):
    tomorrow = date + timedelta(1)
    midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, hour=0, minute=0, second=0)
    return (midnight - date).seconds


num_sec_mid_list = []
energy["nsm"] = ""
for index, row in energy.iterrows():
    date_str = row["date"]
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    num_sec_mid_list.append(cal_num_sec_to_mid(date_obj))

energy["nsm"] = num_sec_mid_list
energy.drop(["rv1", "rv2", "date"], axis=1, inplace=True)

energy.hist(bins=50, figsize=(20, 15))
# plt.show()
# energy.info()"date"
print(energy.corr()['Appliances'].sort_values(ascending=False))
print(energy.describe())

train_set, test_set = train_test_split(energy, test_size=0.2, random_state=42)
x_train = train_set.drop("Appliances", axis=1)
y_train = train_set["Appliances"].copy()

x_test = test_set.drop("Appliances", axis=1)
y_test = test_set["Appliances"].copy()

# Save training data set
np.savetxt("x_train.txt", x_train)
np.savetxt("y_train.txt", y_train)

# Save Testing data set
np.savetxt("x_test.txt", x_test)
np.savetxt("y_test.txt", y_test)
