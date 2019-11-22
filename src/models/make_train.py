# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from src.models.train_model import train_k_and_radius, train_knn_grouping


def main():
    """ Train best 3 models found in notebooks with both training data and
    combined training and validation data. Saves pickled models to ./models
    """
    logger = logging.getLogger(__name__)
    logger.info("Training best models")

    logger.info("training k and radius model with training data")
    train_k_and_radius(
        os.path.join("data", "processed", "train.csv"),
        metric="manhattan",
        weights="uniform",
        n_neighbors=3,
        radius=2,
    )

    logger.info("training knn grouping model (can only use training data)")
    train_knn_grouping(
        os.path.join("data", "processed", "train.csv"),
        metric="euclidean",
        n_neighbors=2,
        weights="squared_distance",
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
