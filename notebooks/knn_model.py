# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True, rc={"figure.figsize": (12.7, 9.27)})

random_state = 123

# %%

df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
df = df.drop(columns=["train"])

# %%

X = df.drop(columns=["longitude", "latitude", "buildingid", "floor"])
y_lon = df.longitude
y_lat = df.latitude
y_building = df.buildingid
y_floor = df.floor

# %%

X_train, X_test, y_train_lon, y_test_lon = train_test_split(
    X,
    y_lon,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_lat, y_test_lat = train_test_split(
    X,
    y_lat,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_building, y_test_building = train_test_split(
    X,
    y_building,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_floor, y_test_floor = train_test_split(
    X,
    y_floor,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

y_train = pd.DataFrame(
    {
        "lon": y_train_lon,
        "lat": y_train_lat,
        "building": y_train_building,
        "floor": y_train_floor,
    }
)

y_test = pd.DataFrame(
    {
        "lon": y_test_lon,
        "lat": y_test_lat,
        "building": y_test_building,
        "floor": y_test_floor,
    }
)

# %%


def calculate_distance(y, y_pred):
    pred_lon = y_pred[:, 0]
    pred_lat = y_pred[:, 1]
    pred_floor = y_pred[:, 3]

    lon_diff2 = (pred_lon - y.lon) ** 2
    lat_diff2 = (pred_lat - y.lat) ** 2
    # lets assume that the height of the floors is 5 meters
    floor_diff2 = ((pred_floor - y.floor) * 5) ** 2

    distance_squared = lon_diff2 + lat_diff2 + floor_diff2

    mean_distance = distance_squared.apply(lambda x: x ** (1 / 2)).mean()

    return mean_distance


distance_scorer = make_scorer(calculate_distance, greater_is_better=False)


# %%

knn_model_no_scaling = Pipeline([("knn", KNeighborsRegressor())])
knn_model_standard_scaler = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]
)
knn_model_robust_scaler = Pipeline(
    [("scaler", RobustScaler()), ("knn", KNeighborsRegressor())]
)


# %% Optimizing hyperparameters

models = [knn_model_no_scaling, knn_model_standard_scaler, knn_model_robust_scaler]
model_names = ["no_scaling", "standard_scaler", "robust_scaler"]

param_grid = {
    "knn__n_neighbors": [1, 2, 3],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
}

results = {}
for name, model in zip(model_names, models):
    param_search = GridSearchCV(
        model, param_grid, scoring=distance_scorer, n_jobs=-2, cv=8, verbose=2
    )

    param_search.fit(X_train, y_train)
    print(name)
    print("Best Params:")
    print(param_search.best_params_)
    print("Best CV Score:")
    print(-param_search.best_score_)

# no_scaling
# Best Params:
# {'knn__n_neighbors': 1, 'knn__p': 1, 'knn__weights': 'uniform'}
# Best CV Score:
# 3.5139507486916415

# standard_scaler
# Best Params:
# {'knn__n_neighbors': 1, 'knn__p': 1, 'knn__weights': 'uniform'}
# Best CV Score:
# 3.8178876723905137

# robust_scaler
# Best Params:
# {'knn__n_neighbors': 1, 'knn__p': 1, 'knn__weights': 'uniform'}
# Best CV Score:
# 3.5139507486916415

# %%

knn_model_no_scaling.fit(X_train, y_train)

# %%

knn_model_standard_scaler.fit(X_train, y_train)

# %%

knn_model_robust_scaler.fit(X_train, y_train)

# %%


# %%

knn_model_no_scaling_score = calculate_distance(
    knn_model_no_scaling, X_test=X_test, y_test=y_test
)
knn_model_standard_scaler_score = calculate_distance(
    knn_model_standard_scaler, X_test=X_test, y_test=y_test
)
knn_model_robust_scaler_score = calculate_distance(
    knn_model_robust_scaler, X_test=X_test, y_test=y_test
)

# %%

print(knn_model_no_scaling_score)
print(knn_model_standard_scaler_score)
print(knn_model_robust_scaler_score)

# %%

pred = knn_model.predict(X_test)
pred_lon = pred[:, 0]
pred_lat = pred[:, 1]
pred_building = pred[:, 2]
pred_floor = pred[:, 3]

predictions = pd.DataFrame(
    {"LATITUDE": pred_lat, "LONGITUDE": pred_lon, "FLOOR": pred_floor}
)

true_values = pd.DataFrame(
    {"LATITUDE": y_test_lat, "LONGITUDE": y_test_lon, "FLOOR": y_test_floor}
)

# %%

for floor in sorted(predictions.FLOOR.unique()):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        data=predictions[predictions["FLOOR"] == int(floor)],
        ax=ax,
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Predictions Floor {int(floor)}")
    plt.show()

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        data=true_values[true_values["FLOOR"] == int(floor)],
        ax=ax,
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Real Values Floor {int(floor)}")
    plt.show()


# %%
