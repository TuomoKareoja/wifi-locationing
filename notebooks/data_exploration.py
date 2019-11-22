# %% Importing and setting styles

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
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

# # Check for Duplicates
#
# * There are 637 duplicated observations in the training set. Removing these
# * No duplicates in the validation set

# %%

print("Number of duplicates in the training set", data_train.duplicated().sum())
data_train = data_train[~data_train.duplicated()]
print("Number of duplicates after cleaning", data_train.duplicated().sum())
print("Number of duplicates in the validation set", data_test.duplicated().sum())


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
    data_train[wap_columns].replace(100, -105).sum(axis=1)
)

for column in random.sample(wap_columns, k=5):
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    sns.scatterplot(
        y="LATITUDE",
        x="LONGITUDE",
        hue=data_train[column].replace(100, np.nan),
        data=data_train,
    )
    plt.title(column)
    plt.axis("equal")
    plt.show()

# %% {markdown}

# # Users

# %% User count and observations per user

print("Number of unique users in the training data:", data_train.USERID.nunique())
data_train.USERID.value_counts().plot(kind="bar")


# %% Signal distribution per user

for user in sorted(data_train.USERID.drop_duplicates()):
    sns.kdeplot(data_train[(data_train.USERID == user)].connection_strength, label=user)
plt.xlim(-55000, -52000)
plt.show()

for user in sorted(data_train.USERID.drop_duplicates()):
    sns.kdeplot(data_train[(data_train.USERID == user)].connection_strength, label=user)
    plt.title(f"User Number {user}")
    plt.xlim(-55000, -52000)
    plt.show()

# Users 7 and 16 seem to have comparatively high values in the right tail

# %% Signal distribution per user and building

for building in sorted(data_train.BUILDINGID.drop_duplicates()):
    for user in sorted(data_train.USERID.drop_duplicates()):
        sns.kdeplot(
            data_train[
                (data_train.USERID == user) & (data_train.BUILDINGID == building)
            ].connection_strength,
            label=user,
        )
        plt.title(building)
    plt.show()

# The tail values that we saw with students are all in building 1

# %% Where are the most extreme measures located?

for building in sorted(data_train.BUILDINGID.drop_duplicates()):
    sns.set(rc={"figure.figsize": (14.7, 11.27)})
    sns.scatterplot(
        y="LATITUDE",
        x="LONGITUDE",
        size="connection_strength",
        hue="connection_strength",
        style="USERID",
        markers=(
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
            "o",
            "v",
        ),
        sizes=(40, 200),
        data=data_train[data_train.BUILDINGID == building],
    )
    plt.title(f"Building {building}")
    plt.axis("equal")
    plt.show()

# %% Who made the observations

for building in sorted(data_train.BUILDINGID.drop_duplicates()):
    sns.set(rc={"figure.figsize": (14.7, 11.27)})
    sns.scatterplot(
        y="LATITUDE",
        x="LONGITUDE",
        hue="USERID",
        style="USERID",
        markers=(
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
            "o",
            "v",
        ),
        s=150,
        alpha=0.2,
        palette=sns.color_palette(
            "husl", data_train[data_train.BUILDINGID == building].USERID.nunique()
        ),
        data=data_train[data_train.BUILDINGID == building],
    )
    plt.title(f"Building {building}")
    plt.axis("equal")
    plt.show()

# The high values seem to be part of pattern and don't seem to be out of place

# %% Trying to find outliers with LOF

lof = LocalOutlierFactor(n_neighbors=3, n_jobs=3)
outliers = lof.fit_predict(X=data_train[wap_columns])

data_train["outlier"] = np.where(outliers == -1, 1, 0)

# %% The percent of observations outliers by user

print("outlier rate:", data_train["outlier"].mean())
data_train.groupby(["USERID"])["outlier"].mean().plot(kind="bar")


# %% Outliers from local outlier factor by building

for building in sorted(data_train.BUILDINGID.drop_duplicates()):
    sns.set(rc={"figure.figsize": (14.7, 11.27)})
    sns.scatterplot(
        y="LATITUDE",
        x="LONGITUDE",
        hue="USERID",
        style="USERID",
        markers=(
            "o",
            "v",
            "^",
            "<",
            ">",
            "8",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
            "o",
            "v",
        ),
        palette=sns.color_palette(
            "husl",
            data_train[
                (data_train.BUILDINGID == building) & (data_train.outlier == 1)
            ].USERID.nunique(),
        ),
        s=150,
        data=data_train[
            (data_train.BUILDINGID == building) & (data_train.outlier == 1)
        ],
    )
    plt.title(f"Building {building}")
    plt.axis("equal")
    plt.show()


# %% operating systems

# coding the os from the information from dataset_info
os_dict = {
    0: "4.0.4",
    1: "2.3.8",
    2: "4.1.0",
    3: "4.0.5",
    4: "4.1.0",
    5: "4.2.0",
    6: "2.3.7",
    7: "2.3.6",
    8: "4.2.2",
    9: "4.3",
    10: "2.3.5",
    11: "4.1.2",
    12: "4.2.0",
    13: "2.3.5",
    14: "4.0.4",
    15: "4.1.0",
    16: "4.0.3",
    17: "4.0.4",
    18: "2.3.4",
    19: "4.2.6",
    20: "4 4.0",
    21: "4.1.0",
    22: "2.3.5",
    23: "4.0.2",
    24: "4.1.1",
}

data_train["os"] = data_train["PHONEID"].replace(os_dict)
data_train["os_base"] = data_train["os"].str[0:3]


# %%

print("outlier rate:", data_train["outlier"].mean())
data_train.groupby(["os"])["outlier"].mean().plot(kind="bar")
plt.title("Outlier rate by OS")
plt.show()
data_train.groupby(["os_base"])["outlier"].mean().plot(kind="bar")
plt.title("Outlier rate by OS")
plt.show()

# %%

data_train.groupby(["USERID", "os"])["outlier"].mean().plot(kind="bar")

# %% Signal distribution

wap_melt = pd.melt(
    data_train,
    id_vars=["USERID", "PHONEID", "BUILDINGID", "FLOOR"],
    value_vars=wap_columns,
)

# change wap to numeric
wap_melt["variable"] = wap_melt["variable"].str.replace("WAP", "").astype(int)

# %%

sns.scatterplot(
    x="variable",
    y="value",
    alpha=0.1,
    data=wap_melt[wap_melt["value"] != 100].sample(100000),
    palette="gist_rainbow",
)
plt.show()

# %%

sns.scatterplot(
    x="variable",
    y="value",
    hue="USERID",
    alpha=0.1,
    data=wap_melt[wap_melt["value"] != 100].sample(100000),
    palette="gist_rainbow",
)
plt.show()
# %%


sns.scatterplot(
    x="variable",
    y="value",
    hue="PHONEID",
    alpha=0.1,
    data=wap_melt[wap_melt["value"] != 100].sample(100000),
    palette="gist_rainbow",
)
plt.show()
# %%


sns.scatterplot(
    x="variable",
    y="value",
    hue="BUILDINGID",
    alpha=0.1,
    data=wap_melt[wap_melt["value"] != 100].sample(100000),
    palette="gist_rainbow",
)
plt.show()

# %%

sns.scatterplot(
    x="variable",
    y="value",
    hue="FLOOR",
    alpha=0.1,
    data=wap_melt[wap_melt["value"] != 100].sample(100000),
    palette="gist_rainbow",
)
plt.show()

# %%

columns = ["PHONEID", "BUILDINGID", "USERID", "FLOOR"]

for column in columns:
    for unique_value in wap_melt[column].drop_duplicates():
        sns.kdeplot(
            wap_melt[(wap_melt["value"] != 100) & (wap_melt[column] == unique_value)][
                "value"
            ],
            label=unique_value,
        )
    plt.legend()
    plt.title(column)
    plt.show()


# %%


# %%

