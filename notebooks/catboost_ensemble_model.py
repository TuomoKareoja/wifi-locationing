# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from IPython.core.interactiveshell import InteractiveShell
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %%

df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
df = df.drop(columns=["train", "relativeposition", "spaceid"])

df_valid = pd.read_csv(os.path.join("data", "processed", "test.csv"))
df_valid = df_valid.drop(columns=["train", "relativeposition", "spaceid"])


# %%

X = df.drop(columns=["longitude", "latitude", "buildingid", "floor"])
y_lon = df.longitude
y_lat = df.latitude
y_building = df.buildingid
y_floor = df.floor

X_valid = df_valid.drop(columns=["longitude", "latitude", "buildingid", "floor"])
y_valid_lon = df_valid.longitude
y_valid_lat = df_valid.latitude
y_valid_building = df_valid.buildingid
y_valid_floor = df_valid.floor

# %% level 1 models

catboost_lon_level1 = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

catboost_lat_level1 = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

catboost_building_level1 = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
)

catboost_floor_level1 = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
)

# %% Longitude

catboost_lon_level1.fit(X, y_lon)

# %% Latitude

catboost_lat_level1.fit(X, y_lat)

# %% Buildings

catboost_building_level1.fit(X, y_building)

# %% Floors

catboost_floor_level1.fit(X, y_floor, eval_set=(X, y_floor))

# %% Predicting with models

pred_lon = catboost_lon_level1.predict(X)
pred_lat = catboost_lat_level1.predict(X)
pred_buildings = catboost_building_level1.predict_proba(X)
pred_floors = catboost_floor_level1.predict_proba(X)

# untangling predictions for different classes

# not sure which predictions refers to which building and floor
pred_building0 = pred_buildings[:, 0]
pred_building1 = pred_buildings[:, 1]
pred_building2 = pred_buildings[:, 2]
pred_floor0 = pred_floors[:, 0]
pred_floor1 = pred_floors[:, 1]
pred_floor2 = pred_floors[:, 2]
pred_floor3 = pred_floors[:, 3]
pred_floor4 = pred_floors[:, 4]


pred_valid_lon = catboost_lon_level1.predict(X_valid)
pred_valid_lat = catboost_lat_level1.predict(X_valid)
pred_valid_buildings = catboost_building_level1.predict_proba(X_valid)
pred_valid_floors = catboost_floor_level1.predict_proba(X_valid)

# untangling predictions for different classes

# not sure which predictions refers to which building and floor
pred_valid_building0 = pred_valid_buildings[:, 0]
pred_valid_building1 = pred_valid_buildings[:, 1]
pred_valid_building2 = pred_valid_buildings[:, 2]
pred_valid_floor0 = pred_valid_floors[:, 0]
pred_valid_floor1 = pred_valid_floors[:, 1]
pred_valid_floor2 = pred_valid_floors[:, 2]
pred_valid_floor3 = pred_valid_floors[:, 3]
pred_valid_floor4 = pred_valid_floors[:, 4]

# %% Creating a new training sets from the predictions

X_comb = pd.DataFrame(
    {
        "lon": pred_lon,
        "lat": pred_lat,
        "building0": pred_building0,
        "building1": pred_building1,
        "building2": pred_building2,
        "floor0": pred_floor0,
        "floor1": pred_floor1,
        "floor2": pred_floor2,
        "floor3": pred_floor3,
        "floor4": pred_floor4,
    }
)

# giving the second level models the predictions of other models
X_lon = pd.concat([X, X_comb], axis="columns").drop(columns=["lon"])
X_lat = pd.concat([X, X_comb], axis="columns").drop(columns=["lat"])
X_building = pd.concat([X, X_comb], axis="columns").drop(
    columns=["building0", "building1", "building2"]
)
X_floor = pd.concat([X, X_comb], axis="columns").drop(
    columns=["floor0", "floor1", "floor2", "floor3", "floor4"]
)

X_valid_comb = pd.DataFrame(
    {
        "lon": pred_valid_lon,
        "lat": pred_valid_lat,
        "building0": pred_valid_building0,
        "building1": pred_valid_building1,
        "building2": pred_valid_building2,
        "floor0": pred_valid_floor0,
        "floor1": pred_valid_floor1,
        "floor2": pred_valid_floor2,
        "floor3": pred_valid_floor3,
        "floor4": pred_valid_floor4,
    }
)

# giving the second level models the predictions of other models
X_valid_lon = pd.concat([X_valid, X_valid_comb], axis="columns").drop(columns=["lon"])
X_valid_lat = pd.concat([X_valid, X_valid_comb], axis="columns").drop(columns=["lat"])
X_valid_building = pd.concat([X_valid, X_valid_comb], axis="columns").drop(
    columns=["building0", "building1", "building2"]
)
X_valid_floor = pd.concat([X_valid, X_valid_comb], axis="columns").drop(
    columns=["floor0", "floor1", "floor2", "floor3", "floor4"]
)

# %% level 2 models

catboost_lon_level2 = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

catboost_lat_level2 = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

catboost_building_level2 = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
)

catboost_floor_level2 = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
)

# %% Longitude

catboost_lon_level2.fit(X_lon, y_lon)

# %% Latitude

catboost_lat_level2.fit(X_lat, y_lat)

# %% Buildings

catboost_building_level2.fit(X_building, y_building)

# %% Floors

catboost_floor_level2.fit(X_floor, y_floor)


# %% Predicting with models

pred_lon = catboost_lon_level2.predict(X_lon)
pred_lat = catboost_lat_level2.predict(X_lat)
pred_buildings = catboost_building_level2.predict_proba(X_building)
pred_floors = catboost_floor_level2.predict_proba(X_floor)

# untangling predictions for different classes

# not sure which predictions refers to which building and floor
pred_building0 = pred_buildings[:, 0]
pred_building1 = pred_buildings[:, 1]
pred_building2 = pred_buildings[:, 2]
pred_floor0 = pred_floors[:, 0]
pred_floor1 = pred_floors[:, 1]
pred_floor2 = pred_floors[:, 2]
pred_floor3 = pred_floors[:, 3]
pred_floor4 = pred_floors[:, 4]

pred_valid_lon = catboost_lon_level2.predict(X_valid_lon)
pred_valid_lat = catboost_lat_level2.predict(X_valid_lat)
pred_valid_buildings = catboost_building_level2.predict_proba(X_valid_building)
pred_valid_floors = catboost_floor_level2.predict_proba(X_valid_floor)

# untangling predictions for different classes

# not sure which predictions refers to which building and floor
pred_valid_building0 = pred_valid_buildings[:, 0]
pred_valid_building1 = pred_valid_buildings[:, 1]
pred_valid_building2 = pred_valid_buildings[:, 2]
pred_valid_floor0 = pred_valid_floors[:, 0]
pred_valid_floor1 = pred_valid_floors[:, 1]
pred_valid_floor2 = pred_valid_floors[:, 2]
pred_valid_floor3 = pred_valid_floors[:, 3]
pred_valid_floor4 = pred_valid_floors[:, 4]


# %% Creating a new training set from the predictions

X_comb = pd.DataFrame(
    {
        "lon": pred_lon,
        "lat": pred_lat,
        "building0": pred_building0,
        "building1": pred_building1,
        "building2": pred_building2,
        "floor0": pred_floor0,
        "floor1": pred_floor1,
        "floor2": pred_floor2,
        "floor3": pred_floor3,
        "floor4": pred_floor4,
    }
)

X_valid_comb = pd.DataFrame(
    {
        "lon": pred_valid_lon,
        "lat": pred_valid_lat,
        "building0": pred_valid_building0,
        "building1": pred_valid_building1,
        "building2": pred_valid_building2,
        "floor0": pred_valid_floor0,
        "floor1": pred_valid_floor1,
        "floor2": pred_valid_floor2,
        "floor3": pred_valid_floor3,
        "floor4": pred_valid_floor4,
    }
)

# %% New combination models

catboost_lon_comb = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

catboost_lat_comb = CatBoostRegressor(
    loss_function="RMSE", eval_metric="RMSE", random_state=random_state
)

# no need to predict the building
catboost_floor_comb = CatBoostClassifier(
    loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
)

# %% Longitude combination model

catboost_lon_comb.fit(X_comb, y_lon)

# %% Latitude combination model

catboost_lat_comb.fit(X_comb, y_lat)

# %% Floor combination model

catboost_floor_comb.fit(X_comb, y_floor)

# %% Predicting with combination models

pred_lon = catboost_lon_comb.predict(X_valid_comb)
pred_lat = catboost_lat_comb.predict(X_valid_comb)
pred_floor = np.hstack(catboost_floor_comb.predict(X_valid_comb))

lon_diff2 = (pred_lon - y_valid_lon) ** 2
lat_diff2 = (pred_lat - y_valid_lat) ** 2
# lets assume that the height of the floors is 5 meters
floor_diff2 = ((pred_floor - y_valid_floor) * 5) ** 2

distance_squared = lon_diff2 + lat_diff2 + floor_diff2

distance = distance_squared.apply(lambda x: x ** (1 / 2))

score = distance.mean()


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
        "LATITUDE": y_valid_lat,
        "LONGITUDE": y_valid_lon,
        "FLOOR": y_valid_floor,
        "distance": distance,
    }
)


# %%

print(f"Mean error in meters {score}")

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

# %% distribution of the error

predictions.distance.hist(bins=100)

# %%
