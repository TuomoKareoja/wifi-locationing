# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor


def load_data(path, group=False):

    df = pd.read_csv(path)

    if "train" in df.columns:
        df.drop(columns=["train"], inplace=True)

    if group:
        df = df.groupby(
            ["buildingid", "floor", "spaceid", "relativeposition"], as_index=False
        ).mean()

    df.drop(columns=["spaceid", "relativeposition"], inplace=True)

    X = df.drop(columns=["longitude", "latitude", "floor", "buildingid"])
    y = pd.DataFrame(
        {
            "lon": df.longitude,
            "lat": df.latitude,
            "floor": df.floor,
            "building": df.buildingid,
        }
    )

    return X, y


def train_knn_grouping(train_data_path, metric, n_neighbors, weights):

    X, y = load_data(train_data_path, group=True)

    if weights == "squared_distance":
        weights = squared_distance

    params = {"metric": metric, "weights": weights, "n_neighbors": n_neighbors}

    model = KNeighborsRegressor(**params)
    model.fit(X, y)

    return model


def train_k_and_radius(
    train_data_path, metric, weights, n_neighbors, radius, extra_data_path=None
):

    params = {
        "metric": metric,
        "weights": weights,
        "n_neighbors": n_neighbors,
        "radius": radius,
    }

    X, y = load_data(train_data_path)
    if extra_data_path:
        X_valid, y_valid = load_data(extra_data_path)
        X = pd.concat([X, X_valid])
        y = pd.concat([y, y_valid])

    model = kradius(**params)
    model.fit(X, y)

    return model


# Implementing a mix of k-kmeans and radius means by modifying sklearn
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


def squared_distance(weights):
    # replacing zero values with machine epsilon
    weights[weights == 0] = np.finfo(float).eps
    weights = [
        (1 / weights_obs ** 2) / np.sum(1 / weights_obs ** 2) for weights_obs in weights
    ]
    return weights


def train_and_save_catboost_ensemble(train_data_path, model_save_path, random_state):

    X, y = load_data(train_data_path)

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

    catboost_lon_level1.fit(X, y.lon)
    catboost_lat_level1.fit(X, y.lat)
    catboost_building_level1.fit(X, y.building)
    catboost_floor_level1.fit(X, y.floor)

    pred_lon = catboost_lon_level1.predict(X)
    pred_lat = catboost_lat_level1.predict(X)
    pred_buildings = catboost_building_level1.predict_proba(X)
    pred_floors = catboost_floor_level1.predict_proba(X)

    X_comb = combine_predictions(pred_lon, pred_lat, pred_buildings, pred_floors)

    X_lon = pd.concat([X, X_comb], axis="columns").drop(columns=["lon"])
    X_lat = pd.concat([X, X_comb], axis="columns").drop(columns=["lat"])
    X_building = pd.concat([X, X_comb], axis="columns").drop(
        columns=["building0", "building1", "building2"]
    )
    X_floor = pd.concat([X, X_comb], axis="columns").drop(
        columns=["floor0", "floor1", "floor2", "floor3", "floor4"]
    )

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

    catboost_lon_level2.fit(X_lon, y.lon)
    catboost_lat_level2.fit(X_lat, y.lat)
    catboost_building_level2.fit(X_building, y.building)
    catboost_floor_level2.fit(X_floor, y.floor)

    pred_lon = catboost_lon_level2.predict(X_lon)
    pred_lat = catboost_lat_level2.predict(X_lat)
    pred_buildings = catboost_building_level2.predict_proba(X_building)
    pred_floors = catboost_floor_level2.predict_proba(X_floor)

    X_comb = combine_predictions(pred_lon, pred_lat, pred_buildings, pred_floors)

    catboost_lon_comb = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE", random_state=random_state
    )

    catboost_lat_comb = CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE", random_state=random_state
    )

    catboost_floor_comb = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
    )

    catboost_building_comb = CatBoostClassifier(
        loss_function="MultiClass", eval_metric="MultiClass", random_state=random_state
    )

    catboost_lon_comb.fit(X_comb, y.lon)
    catboost_lat_comb.fit(X_comb, y.lat)
    catboost_floor_comb.fit(X_comb, y.floor)
    catboost_building_comb.fit(X_comb, y.building)

    model_dict = {
        "catboost_lon_level1": catboost_lon_level1,
        "catboost_lat_level1": catboost_lat_level1,
        "catboost_building_level1": catboost_building_level1,
        "catboost_floor_level1": catboost_floor_level1,
        "catboost_lon_level2": catboost_lon_level2,
        "catboost_lat_level2": catboost_lat_level2,
        "catboost_building_level2": catboost_building_level2,
        "catboost_floor_level2": catboost_floor_level2,
        "catboost_lon_comb": catboost_lon_comb,
        "catboost_lat_comb": catboost_lat_comb,
        "catboost_building_comb": catboost_building_comb,
        "catboost_floor_comb": catboost_floor_comb,
    }

    pickle.dump(model_dict, open(model_save_path, "wb"))


def combine_predictions(pred_lon, pred_lat, pred_buildings, pred_floors):

    pred_building0 = pred_buildings[:, 0]
    pred_building1 = pred_buildings[:, 1]
    pred_building2 = pred_buildings[:, 2]
    pred_floor0 = pred_floors[:, 0]
    pred_floor1 = pred_floors[:, 1]
    pred_floor2 = pred_floors[:, 2]
    pred_floor3 = pred_floors[:, 3]
    pred_floor4 = pred_floors[:, 4]

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

    return X_comb
