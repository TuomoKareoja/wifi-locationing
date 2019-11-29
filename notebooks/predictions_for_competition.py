# %%

import os

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from src.models.scoring import distance75

# Setting styles
InteractiveShell.ast_node_interactivity = "all"

random_state = 123

KNeighborsRegressor()

# %%

df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
df = df.drop(columns=["spaceid", "relativeposition"])

df_valid = pd.read_csv(os.path.join("data", "processed", "test.csv"))
df_valid = df_valid.drop(columns=["spaceid", "relativeposition"])

# %%

df_comp = pd.read_csv(os.path.join("data", "processed", "comp.csv"))
df_comp = df_comp.drop(columns=["spaceid", "relativeposition"])


# %% [markdown]

# # Comparing which columns have no signal between different datasets
#
# * We use only WAPs that have signal in the competition dataset for training
# * WAPs also have to have a signal in training or validation sets

# %%


wap_columns = [column for column in df.columns if "wap" in column]

# %%

no_signal_train = [
    column for column in wap_columns if pd.Series(df[column] == -110).all()
]
no_signal_test = [
    column for column in wap_columns if pd.Series(df_valid[column] == -110).all()
]
no_signal_comp = [
    column for column in wap_columns if pd.Series(df_comp[column] == -110).all()
]

# %%

# find all WAPs that have no signal in training or test set (full training data)
no_signal_all_train_test = set(no_signal_train).intersection(set(no_signal_test))
# combine previous to columns that have no signal in the competition data
no_signal_columns = set(no_signal_comp).union(no_signal_all_train_test)


# %%

# dropping non suitable WAP signal

df.drop(columns=no_signal_columns, inplace=True)
df_valid.drop(columns=no_signal_columns, inplace=True)
df_comp.drop(columns=no_signal_columns, inplace=True)

# %%

# combining training and test data to full training set

df = pd.concat([df, df_valid])
df.drop(columns="train", inplace=True)

# %% Putting the data to suitable format for sklearn

X = df.drop(columns=["longitude", "latitude", "floor", "buildingid"])
y = pd.DataFrame(
    {
        "lon": df.longitude,
        "lat": df.latitude,
        "floor": df.floor,
        "building": df.buildingid,
    }
)

X_comp = df_comp.drop(columns=["longitude", "latitude", "floor", "buildingid"])

# %% Modifying KNN regressor to also use a radius filter

# Beware!!! Really flimsy and hacky and does not work for weights
# that are not uniform or a custom distance squared


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

            _y = self.knn_model._y
            if _y.ndim == 1:
                _y = _y.reshape((-1, 1))

            if weights is None:
                y_pred = np.array([np.mean(_y[ind, :], axis=0) for ind in inds])

            else:
                y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)

                for k in range(X.shape[0]):
                    for j in range(_y.shape[1]):
                        y_pred[k, j] = np.sum(
                            _y[inds[k], j] * weights[k] / sum(weights[k])
                        )

            if _y.ndim == 1:
                y_pred = y_pred.ravel()

            return y_pred

        else:
            return self.knn_model.predict(X)


# %% making a scorer function


def calculate_distance(y, y_pred):
    distance = distance75(y, y_pred)
    return np.mean(distance)


distance_scorer = make_scorer(calculate_distance, greater_is_better=False)

# %% new weight metric


def squared_distance(weights):
    # replacing zero values with machine epsilon
    weights = [
        np.array([1 / max(weight, np.finfo(float).eps) ** 2 for weight in weights_obs])
        for weights_obs in weights
    ]
    return weights


# %% Initial hyperparameter search

metric_opt = ["manhattan"]
weights_opt = ["uniform", squared_distance]
n_neighbors_opt = [2, 3, 5]
radius_opt = [1, 4, 12, 25]

best_score = None
for n_neighbors in n_neighbors_opt:
    for weights in weights_opt:
        for metric in metric_opt:
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
                folds = KFold(n_splits=8, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(
                    kradius_model, X, y, scoring=distance_scorer, cv=folds, n_jobs=-2
                )
                score = -np.mean(cv_scores)
                print(
                    f"metric: {metric}, weight: {weights},",
                    f"n_neighbors: {n_neighbors}, radius: {radius},",
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

# {'metric': 'manhattan', 'weights': 'uniform', 'n_neighbors': 5, 'radius': 1}


# %% second hyperparameter pass

metric_opt = ["manhattan"]
weights_opt = ["uniform", squared_distance]
n_neighbors_opt = [4]
radius_opt = [1, 2, 3, 4]

best_score = None
for n_neighbors in n_neighbors_opt:
    for weights in weights_opt:
        for metric in metric_opt:
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
                folds = KFold(n_splits=8, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(
                    kradius_model, X, y, scoring=distance_scorer, cv=folds, n_jobs=-2
                )
                score = -np.mean(cv_scores)
                print(
                    f"metric: {metric}, weight: {weights},",
                    f"n_neighbors: {n_neighbors}, radius: {radius},",
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

# {'metric': 'manhattan', 'weights': 'uniform', 'n_neighbors': 4, 'radius': 1}

# %% third hyperparameter pass

metric_opt = ["manhattan"]
weights_opt = ["uniform"]
n_neighbors_opt = [1, 4, 5]
radius_opt = [0.8, 1, 1.2, 1.8, 2, 2.2]

best_score = None
for n_neighbors in n_neighbors_opt:
    for weights in weights_opt:
        for metric in metric_opt:
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
                folds = KFold(n_splits=12, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(
                    kradius_model, X, y, scoring=distance_scorer, cv=folds, n_jobs=-2
                )
                score = -np.mean(cv_scores)
                print(
                    f"metric: {metric}, weight: {weights},",
                    f"n_neighbors: {n_neighbors}, radius: {radius},",
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

# {'metric': 'manhattan', 'weights': 'uniform', 'n_neighbors': 5, 'radius': 1}

# %% last hyperparameter pass

metric_opt = ["manhattan"]
weights_opt = ["uniform"]
n_neighbors_opt = [4, 5]
radius_opt = [0.9, 0.95, 1, 1.05, 1.1]

best_score = None
for n_neighbors in n_neighbors_opt:
    for weights in weights_opt:
        for metric in metric_opt:
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
                folds = KFold(n_splits=20, shuffle=True, random_state=random_state)
                cv_scores = cross_val_score(
                    kradius_model, X, y, scoring=distance_scorer, cv=folds, n_jobs=-2
                )
                score = -np.mean(cv_scores)
                print(
                    f"metric: {metric}, weight: {weights},",
                    f"n_neighbors: {n_neighbors}, radius: {radius},",
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

# {'metric': 'manhattan', 'weights': 'uniform', 'n_neighbors': 5, 'radius': 1}

# %% comparison to simple knn

metric_opt = ["manhattan"]
weights_opt = ["uniform", "distance", squared_distance]
n_neighbors_opt = [1, 2, 3, 5]

best_score_simple_knn = None
for n_neighbors in n_neighbors_opt:
    # no need to check different weights if k = 1
    if n_neighbors == 1:
        weights_opt_modified = ["uniform"]
    else:
        weights_opt_modified = weights_opt.copy()
    for weights in weights_opt_modified:
        for metric in metric_opt:
            knn_model = KNeighborsRegressor(
                metric=metric, weights=weights, n_neighbors=n_neighbors, n_jobs=1
            )
            folds = KFold(n_splits=20, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                knn_model, X, y, scoring=distance_scorer, cv=folds, n_jobs=-2
            )
            score = -np.mean(cv_scores)
            print(
                f"metric: {metric}, weight: {weights},",
                f"n_neighbors: {n_neighbors},",
                f"score: {score}",
            )

            if best_score_simple_knn is None or best_score_simple_knn > score:
                print("new best score found!")
                best_score_simple_knn = score
                best_params_simple_knn = {
                    "metric": metric,
                    "weights": weights,
                    "n_neighbors": n_neighbors,
                }


print(best_params_simple_knn)

# {'metric': 'manhattan', 'weights': <function squared_distance at 0x7f149c7e29d8>, 'n_neighbors': 2}

# %% Training the kradius model with full data and optimized hyperparameters

kradius_model = kradius(**best_params)
kradius_model.fit(X, y)

# %% Training the kradius model with full data and optimized hyperparameters

knn_model = KNeighborsRegressor(**best_params_simple_knn)
knn_model.fit(X, y)

# %% predicting competition results with kradius

kradius_pred = kradius_model.predict(X_comp)
kradius_pred_lon = kradius_pred[:, 0]
kradius_pred_lat = kradius_pred[:, 1]
kradius_pred_floor = np.round(kradius_pred[:, 2], decimals=0)
kradius_pred_building = np.round(kradius_pred[:, 3], decimals=0)

kradius_preds_df = pd.DataFrame(
    {
        "LATITUDE": kradius_pred_lat,
        "LONGITUDE": kradius_pred_lon,
        "FLOOR": kradius_pred_floor,
    }
)

# %% predicting competition results with simple knn

knn_pred = knn_model.predict(X_comp)
knn_pred_lon = knn_pred[:, 0]
knn_pred_lat = knn_pred[:, 1]
knn_pred_floor = np.round(knn_pred[:, 2], decimals=0)
knn_pred_building = np.round(knn_pred[:, 3], decimals=0)

knn_preds_df = pd.DataFrame(
    {"LATITUDE": knn_pred_lat, "LONGITUDE": knn_pred_lon, "FLOOR": knn_pred_floor}
)

# %% saving to disk

kradius_preds_df.to_csv(
    os.path.join("data", "predictions", "tuomo_kradius.csv"), index=False
)

knn_preds_df.to_csv(
    os.path.join("data", "predictions", "tuomo_simple_knn.csv"), index=False
)

# %%
