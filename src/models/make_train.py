# -*- coding: utf-8 -*-
import logging
import os
import pickle
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from src.models.train_model import (
    train_k_and_radius,
    train_knn_grouping,
    train_and_save_catboost_ensemble,
)

random_state = 123


def main():
    """ Train best 3 models found in notebooks with both training data and
    combined training and validation data. Saves pickled models to ./models
    """
    logger = logging.getLogger(__name__)
    logger.info("Training best models")

    logger.info("training k and radius model with training data")
    k_and_radius_model = train_k_and_radius(
        train_data_path=os.path.join("data", "processed", "train.csv"),
        metric="manhattan",
        weights="uniform",
        n_neighbors=3,
        radius=2,
    )
    pickle.dump(
        k_and_radius_model, open(os.path.join("models", "k_and_radius_model.p"), "wb")
    )

    logger.info("training knn grouping model (can only use training data)")
    knn_grouping_model = train_knn_grouping(
        train_data_path=os.path.join("data", "processed", "train.csv"),
        metric="euclidean",
        n_neighbors=2,
        weights="squared_distance",
    )
    pickle.dump(
        knn_grouping_model, open(os.path.join("models", "knn_grouping_model.p"), "wb")
    )

    logger.info("training catboost ensemble model")
    knn_grouping_model = train_and_save_catboost_ensemble(
        train_data_path=os.path.join("data", "processed", "train.csv"),
        model_save_path=os.path.join("models", "catboost_ensemble_model_dict.p"),
        random_state=random_state,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
