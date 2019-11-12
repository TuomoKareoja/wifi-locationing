# %% Importing and setting styles

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

# %% Loading data

raw_path = os.path.join("data", "raw")

data_train = pd.read_csv(os.path.join(raw_path, "trainingData.csv"))
data_test = pd.read_csv(os.path.join(raw_path, "validationData.csv"))

data = pd.concat([data_train, data_test], ignore_index=True)

# %% [markdown]

# * The datasets are not very big, but they have a lot of columns.
# * Train and test set have all the same columns

# %%

print("Train set size:", len(data_train))
print("Test set size:", len(data_test))

data_train.head()
data_test.head()

data_train.dtypes
print("Do train and test datatypes match?", all(data_train.dtypes == data_test.dtypes))


# %% [markdown]

# * There are 520 different WAP columns
# * These columns have no missing values
# * Missing values (lack of any connection) is coded as 100
# and these values are very common for the WiFi stations

# %%

wap_columns = [column for column in data_train.columns if "WAP" in column]
print("the amount of WAP columns:", len(wap_columns))
print("missing values in train set:", sum(data_train[wap_columns].isnull().sum()))
print("missing values in test set:", sum(data_test[wap_columns].isnull().sum()))

is100 = data == 100
data[wap_columns].where(is100).count(axis="rows").hist(bins=100)


# %% [markdown]

# * There are 9 non-WAP columns

# columns BUILDINGID, SPACEID, USERID and PHONEID need to made categorical
# Floor is okay as an integer because it has a natural ordering

# %%

non_wap_columns = [column for column in data_train.columns if "WAP" not in column]
print("the amount of non-WAP columns:", len(non_wap_columns))

data_train[non_wap_columns].isnull().sum()
data_test[non_wap_columns].isnull().sum()

data_train[non_wap_columns].isnull().sum()
data_test[non_wap_columns].isnull().sum()


for column in non_wap_columns:
    data_train[column].value_counts()


# %% [markdown]

# * Users are not defined in the testset so using them as features is not advisable
# * There are phones with ID:s that are not represented in the test set and vice versa.
# This makes this also very hard to use as a feature
# * Timestamp is useless also by the way

# %%

cat_columns = ["BUILDINGID", "SPACEID", "USERID", "PHONEID"]
for column in cat_columns:
    data_train[column].value_counts().plot.bar()
    plt.suptitle("Trainset: " + column)
    plt.show()
    data_test[column].value_counts().plot.bar()
    plt.suptitle("Testset: " + column)
    plt.show()


# %% [markdown]

# the WiFi connection points are usually only noticeable from
# one building, but this is not always the case

# Connection strength does not follow super clear patterns, but the
# farther points usually have a worse connection.

# There are some WiFi connection points that do not have any readings
# in the dataset

# %%

data_train["connection_strength"] = (
    data_train[wap_columns].replace(100, np.nan).sum(axis=1)
)

for column in random.sample(wap_columns, k=5):
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.scatterplot(
        x="LATITUDE",
        y="LONGITUDE",
        hue=data_train[column].replace(100, np.nan),
        data=data_train,
    )
    plt.title(column)
    plt.axis("equal")
    plt.show()

# %% [markdown]

# # Distribution of the signal strength
