# -*- coding: utf-8 -*-
import numpy as np


def distance75(y, y_pred):

    pred_lon = y_pred[:, 0]
    pred_lat = y_pred[:, 1]
    pred_floor = y_pred[:, 2]
    pred_building = y_pred[:, 3]

    # in case the neighbors is > 1 then we need to make sure that floor is int
    pred_floor = np.round(pred_floor, decimals=0)
    pred_building = np.round(pred_building, decimals=0)

    distance75 = (
        np.absolute(pred_lon - y.lon)
        + np.absolute(pred_lat - y.lat)
        + 4 * np.absolute(pred_floor - y.floor)
        + 50 * np.absolute(pred_building - y.building)
    )

    return distance75
