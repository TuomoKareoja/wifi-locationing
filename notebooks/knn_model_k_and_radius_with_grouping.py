# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
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

# %% Implementing a mix of k-kmeans and radius means by modifying sklearn


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``

    Parameters
    ----------
    dist : ndarray
        The input distances
    weights : {'uniform', 'distance' or a callable}
        The kind of weighting used

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        if ``weights == 'uniform'``, then returns None
    """
    if weights in (None, "uniform"):
        return None
    elif weights == "distance":
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                    dist[point_dist_i] = point_dist == 0.0
                else:
                    dist[point_dist_i] = 1.0 / point_dist
        else:
            with np.errstate(divide="ignore"):
                dist = 1.0 / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError(
            "weights not recognized: should be 'uniform', "
            "'distance', or a callable function"
        )


class kradius(BaseEstimator):
    def __init__(
        self, metric="euclidean", weights="uniform", n_neighbors=3, radius=1.0, n_jobs=1
    ):
        self.radius = radius
        self.metric = metric
        self.weights = weights
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            weights=self.weights,
            metric=self.metric,
        )
        self.knn_model.fit(X, y)
        return self

    def predict(self, X):

        # no need to to distance filtering if we take only 1 neighbor
        if self.n_neighbors > 1:

            _y = self.knn_model._y

            dists, inds = self.knn_model.kneighbors(X, n_neighbors=self.n_neighbors)
            # dropping value where distance too big
            # we always keep the closest point (first value)
            inds = [
                np.array(
                    [
                        index
                        for distance, index in zip(dist, ind)
                        if distance <= self.radius or distance == dist[0]
                    ]
                )
                for dist, ind in zip(dists, inds)
            ]
            dists = [
                np.array(
                    [
                        distance
                        for distance in dist
                        if distance <= self.radius or distance == dist[0]
                    ]
                )
                for dist in dists
            ]

            weights = _get_weights(dists, self.weights)

            empty_obs = np.full_like(_y[0], np.nan)

            if weights is None:
                y_pred = np.array(
                    [
                        np.mean(_y[ind, :], axis=0) if len(ind) else empty_obs
                        for (i, ind) in enumerate(inds)
                    ]
                )

            else:
                y_pred = np.array(
                    [
                        np.average(_y[ind, :], axis=0, weights=weights[i])
                        if len(ind)
                        else empty_obs
                        for (i, ind) in enumerate(inds)
                    ]
                )

            return y_pred

        else:
            return self.knn_model.predict(X)


# %% Optimizing hyperparameters


def calculate_distance(y, y_pred):
    distance = distance75(y, y_pred)
    return np.mean(distance)


distance_scorer = make_scorer(calculate_distance, greater_is_better=False)


def squared_distance(weights):
    # replacing zero values with machine epsilon
    weights[weights == 0] = np.finfo(float).eps
    weights = [
        (1 / weights_obs ** 2) / np.sum(1 / weights_obs ** 2) for weights_obs in weights
    ]
    return weights


metric_opt = ["euclidean", "manhattan"]
weights_opt = ["uniform"]
n_neighbors_opt = range(1, 6)
radius_opt = np.arange(120, 200, step=1)

best_score = None
for metric in metric_opt:
    for weights in weights_opt:
        for n_neighbors in n_neighbors_opt:
            # no need to check for are if just one k
            if n_neighbors == 1:
                radius_opt_modified = [1]
            else:
                radius_opt_modified = radius_opt.copy()

            for radius in radius_opt_modified:
                kradius_model = kradius(
                    metric=metric,
                    weights=weights,
                    n_neighbors=n_neighbors,
                    radius=radius,
                    n_jobs=1,
                )
                folds = KFold(n_splits=10, shuffle=True, random_state=random_state)
                cv_val_scores = cross_val_score(
                    kradius_model,
                    X=X,
                    y=y,
                    scoring=distance_scorer,
                    cv=folds,
                    n_jobs=-2,
                )
                score = -cv_val_scores.mean()
                print()
                print(
                    f"metric: {metric},",
                    f"weights: {weights},",
                    f"n_neighbors: {n_neighbors},",
                    f"radius: {radius},",
                    f"score: {score}",
                )
                if best_score is None or best_score > score:
                    print("new best score found!")
                    best_score = score
                    best_params = {
                        "metric": metric,
                        "weights": weights,
                        "n_neighbors": n_neighbors,
                        "radius": radius,
                    }


print(best_params)

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
print(f"Latitude error: {lat_score}")
print(f"Longitude error: {lon_score}")
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
