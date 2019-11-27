# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from src.models.train_model import combine_predictions


def predict_catboost_ensemble(
    X,
    catboost_lon_level1,
    catboost_lat_level1,
    catboost_building_level1,
    catboost_floor_level1,
    catboost_lon_level2,
    catboost_lat_level2,
    catboost_building_level2,
    catboost_floor_level2,
    catboost_lon_comb,
    catboost_lat_comb,
    catboost_building_comb,
    catboost_floor_comb,
):

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

    pred_lon = catboost_lon_level2.predict(X_lon)
    pred_lat = catboost_lat_level2.predict(X_lat)
    pred_buildings = catboost_building_level2.predict_proba(X_building)
    pred_floors = catboost_floor_level2.predict_proba(X_floor)

    X_comb = combine_predictions(pred_lon, pred_lat, pred_buildings, pred_floors)

    pred_lon = catboost_lon_comb.predict(X_comb)
    pred_lat = catboost_lat_comb.predict(X_comb)
    pred_floor = np.hstack(catboost_floor_comb.predict(X_comb))
    pred_building = np.hstack(catboost_building_comb.predict(X_comb))

    predictions = np.column_stack((pred_lon, pred_lat, pred_floor, pred_building))

    return predictions
