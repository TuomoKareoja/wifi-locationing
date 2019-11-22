# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor

from src.models.scoring import distance75

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %%

df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
df = df.drop(columns=["train"])

df_valid = pd.read_csv(os.path.join("data", "processed", "test.csv"))
df_valid = df_valid.drop(columns=["train", "spaceid", "relativeposition"])

# %% grouping the training data by location

# this drops the amount of datapoints by 95 %
df = df.groupby(
    ["buildingid", "floor", "spaceid", "relativeposition"], as_index=False
).mean()

df.drop(columns=["spaceid", "relativeposition"], inplace=True)

# %%

X = df.drop(columns=["longitude", "latitude", "floor", "buildingid"])
y = pd.DataFrame(
    {
        "lon": df.longitude,
        "lat": df.latitude,
        "floor": df.floor,
        "building": df.buildingid,
    }
)

X_valid = df_valid.drop(columns=["longitude", "latitude", "floor", "buildingid"])
y_valid = pd.DataFrame(
    {
        "lon": df_valid.longitude,
        "lat": df_valid.latitude,
        "floor": df_valid.floor,
        "building": df_valid.buildingid,
    }
)

# %%


def calculate_distance(y, y_pred):
    distance = distance75(y, y_pred)
    return np.mean(distance)


distance_scorer = make_scorer(calculate_distance, greater_is_better=False)


# %% Optimizing hyperparameters


def squared_distance(weights):
    # replacing zero values with machine epsilon
    weights[weights == 0] = np.finfo(float).eps
    weights = [
        (1 / weights_obs ** 2) / np.sum(1 / weights_obs ** 2) for weights_obs in weights
    ]
    return weights


param_grid = {
    "n_neighbors": [1, 2, 3],
    "weights": ["uniform", "distance", squared_distance],
    "metric": ["euclidean", "manhattan"],
}

# there might be some inherent order in the dataset
# so shuffling to get rid of this
folds = KFold(n_splits=10, shuffle=True, random_state=random_state)

knn_model = KNeighborsRegressor()

param_search = GridSearchCV(
    knn_model, param_grid, scoring=distance_scorer, n_jobs=-2, cv=folds, verbose=2
)

param_search.fit(X, y)
print("Best Params:")
print(param_search.best_params_)
print("Best CV Score:")
print(-param_search.best_score_)

best_params = param_search.best_params_

# %% Training the model with full data and optimized hyperparameters

knn_model = KNeighborsRegressor(**best_params)
knn_model.fit(X, y)

pred = knn_model.predict(X_valid)

pred_lon = pred[:, 0]
pred_lat = pred[:, 1]
pred_floor = np.round(pred[:, 2], decimals=0)
pred_building = np.round(pred[:, 3], decimals=0)

distance = distance75(y_valid, pred)
score = np.mean(distance)
lon_score = np.mean(np.absolute(pred_lon - y_valid.lon))
lat_score = np.mean(np.absolute(pred_lat - y_valid.lat))
right_floor = np.round(np.mean(pred_floor == y_valid.floor) * 100, 2)
right_building = np.round(np.mean(pred_building == y_valid.building) * 100, 2)

predictions = pd.DataFrame(
    {
        "LATITUDE": pred_lat,
        "LONGITUDE": pred_lon,
        "FLOOR": pred_floor,
        "distance": distance,
    }
)

true_values = pd.DataFrame(
    {
        "LATITUDE": y_valid.lat,
        "LONGITUDE": y_valid.lon,
        "FLOOR": y_valid.floor,
        "distance": distance,
    }
)

# %%

print(f"Mean error in distance75: {score}")
print(f"Latitude error: {lat_score} %")
print(f"Longitude error: {lon_score} %")
print(f"Floors correct: {right_floor} %")
print(f"Building correct: {right_building} %")


for floor in sorted(predictions.FLOOR.unique()):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        hue="distance",
        ax=ax,
        s=100,
        data=predictions[predictions["FLOOR"] == int(floor)],
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Predictions Floor {int(floor)}")
    plt.show()

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        hue="distance",
        s=100,
        data=true_values[true_values["FLOOR"] == int(floor)],
        ax=ax,
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Real Values Floor {int(floor)}")
    plt.show()


# %% distribution of the errors

predictions.distance.hist(bins=100)

# %%
