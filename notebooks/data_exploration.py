#%% Importing and setting styles

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%% Loading data

raw_path = os.path.join("data", "raw")

data_train = pd.read_csv(os.path.join(raw_path, "trainingData.csv"))
data_test = pd.read_csv(os.path.join(raw_path, "validationData.csv"))

#%% [markdown]

# The datasets are not very big, but they have a lot of columns
# Train and test set have all the same columns?

#%%

print("Train set size:", len(data_train))
print("Test set size:", len(data_test))

data_train.head()
data_test.head()

#%% [markdown]


#%%

data_train.dtypes
print("Do train and test datatypes match?", all(data_train.dtypes == data_test.dtypes))


#%% [markdown]

# Lets look at the columns that start WAP

# 520 different column but no missing values

#%%

wap_columns = [column for column in data_train.columns if "WAP" in column]
print("the amount of WAP columns:", len(wap_columns))
sum(data_train[wap_columns].isnull().sum())
sum(data_train[wap_columns].isnull().sum())

# wap_corr_train = np.corrcoef(data_train[wap_columns])
# sns.heatmap(wap_corr_train)
# wap_corr_test = np.corrcoef(data_test[wap_columns])
# sns.heatmap(wap_corr_test)


#%% [markdown]

# Non-WAP columns are 9

# Don't contain missing values, but 100 is a missing signal

# columns BUILDINGID, SPACEID, USERID and PHONEID need to made categorical
# Floor is okay as an integer because it has a natural ordering

#%%

non_wap_columns = [column for column in data_train.columns if "WAP" not in column]
print("the amount of non-WAP columns:", len(non_wap_columns))

data_train[non_wap_columns].isnull().sum()
data_test[non_wap_columns].isnull().sum()

data_train[non_wap_columns].isnull().sum()
data_test[non_wap_columns].isnull().sum()


for column in non_wap_columns:
    data_train[column].value_counts()

#%%

cat_columns = ["BUILDINGID", "SPACEID", "USERID", "PHONEID"]
for column in cat_columns:
    data_train[column] = data_train[column].astype("category")
    data_test[column] = data_test[column].astype("category")


#%% [markdown]

# Users are not defined in the testset so using them as features is not advisable

# There are phones with ID:s that are not represented in the test set and vice versa.
# This makes this also very hard to use as a feature

# Timestamp is useless also by the way

#%%

cat_columns = ["BUILDINGID", "SPACEID", "USERID", "PHONEID"]
for column in cat_columns:
    data_train[column].value_counts().plot.bar()
    plt.suptitle("Trainset: " + column)
    plt.show()
    data_test[column].value_counts().plot.bar()
    plt.suptitle("Testset: " + column)
    plt.show()


#%% [markdown]

# What is the distribution of signals from wap-columns?

#%%

for column in wap_columns[:100]:
    try:
        sns.kdeplot(data_test[column], color="lightgrey", legend=False)
    except:
        print("Column {} could not be plotted".format(column))

plt.show()


#%% [markdown]

# TODO: Should the coordinates be "straightened" to get good readings?

data_train["connection_strength"] = (
    data_train[wap_columns].replace(100, np.nan).sum(axis=1)
)
#%%

for floor in sorted(data_train["FLOOR"].unique()):

    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.scatterplot(
        x="LATITUDE",
        y="LONGITUDE",
        hue="connection_strength",
        data=data_train,
        alpha=0.5,
    )
    plt.title("Floor number: {}".format(floor))
    plt.show()

#%%

data_train = data_train.assign(
    x=np.cos(data_train.LATITUDE) * np.cos(data_train.LONGITUDE)
)
data_train = data_train.assign(
    y=np.cos(data_train.LATITUDE) * np.sin(data_train.LONGITUDE)
)
data_train = data_train.assign(z=np.sin(data_train.LATITUDE))

#%%

data_train["building"] = data_train["BUILDINGID"].astype(int)
sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.scatterplot(x="x", y="y", hue="building", data=data_train)
plt.show()

#%%
