# -*- coding: utf-8 -*-
import logging
import os
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.models.predict_model import predict_catboost_ensemble


@click.command()
@click.argument("predict_data_path", type=click.Path(exists=True))
def main(predict_data_path):
    """Predict with models saved in ./models
    """
    logger = logging.getLogger(__name__)
    logger.info("Predicting with best models")

    logger.info("Loading in the dataset to predict")
    orig_df = pd.read_csv(predict_data_path)
    X = orig_df.copy()
    # lowercasing columns to fit trained models
    X.columns = [column.lower() for column in X.columns]
    # keeping only signals
    wap_columns = [column for column in X.columns if "wap" in column]
    X = X[wap_columns]

    logger.info("Predicting with k and radius model")
    k_and_radius_model = pickle.load(
        open(os.path.join("models", "k_and_radius_model.p"), "rb")
    )
    k_and_radius_preds = k_and_radius_model.predict(X)

    logger.info("Predicting with knn grouped model")
    knn_grouping_model = pickle.load(
        open(os.path.join("models", "knn_grouping_model.p"), "rb")
    )
    knn_grouping_preds = knn_grouping_model.predict(X)

    logger.info("Predicting with catboost ensemble model")
    catboost_ensemble_model_dict = pickle.load(
        open(os.path.join("models", "catboost_ensemble_model_dict.p"), "rb")
    )
    catboost_ensemble_preds = predict_catboost_ensemble(
        X, **catboost_ensemble_model_dict
    )

    logger.info("Adding predictions to predicted data as new columns")
    for model_name, preds in zip(
        ["k_and_radius", "knn_grouping", "catboost_ensemble"],
        [k_and_radius_preds, knn_grouping_preds, catboost_ensemble_preds],
    ):
        pred_lon = preds[:, 0]
        pred_lat = preds[:, 1]
        pred_floor = np.round(preds[:, 2], decimals=0)
        pred_building = np.round(preds[:, 3], decimals=0)

        orig_df[model_name + "_lon"] = pred_lon
        orig_df[model_name + "_lat"] = pred_lat
        orig_df[model_name + "_floor"] = pred_floor
        orig_df[model_name + "_building"] = pred_building

    logger.info("Saving to data/predictions")
    orig_df.to_csv(os.path.join("data", "predictions", "predictions.csv"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
