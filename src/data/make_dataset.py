# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.data.clean_data import change_projection


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Loading the datasets")
    df_train = pd.read_csv(os.path.join("data", "raw", "trainingData.csv"))
    df_test = pd.read_csv(os.path.join("data", "raw", "validationData.csv"))

    logger.info("Combining the datasets")
    df_train["train"] = True
    df_test["train"] = False
    df = pd.concat([df_train, df_test], ignore_index=True)

    logger.info(
        "Dropping unnecessary columns not available in the validation set (all null)"
    )
    columns_to_drop = ["RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP", "SPACEID"]
    df.drop(columns=columns_to_drop, inplace=True)

    logger.info("Making all columns lowercase for easier typing")
    df.columns = [column.lower() for column in df.columns]

    logger.info("Changing building ID to categorical")
    cat_columns = ["buildingid"]
    for column in cat_columns:
        df[column] = df[column].astype("category")

    logger.info("Saving to data/processed")
    df_train = df[df.train]
    df_test = df[~df.train]
    df_train.to_csv(os.path.join("data", "processed", "train.csv"), index=False)
    df_test.to_csv(os.path.join("data", "processed", "test.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
