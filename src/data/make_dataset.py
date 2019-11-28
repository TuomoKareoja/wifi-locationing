# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Loading the datasets")
    df_train = pd.read_csv(os.path.join("data", "raw", "trainingData.csv"))
    df_test = pd.read_csv(os.path.join("data", "raw", "validationData.csv"))
    df_comp = pd.read_csv(os.path.join("data", "raw", "competitionData.csv"))

    logger.info("Combining the datasets")
    df_train["train"] = True
    df_test["train"] = False
    df_comp["train"] = False
    df_train["comp"] = False
    df_test["comp"] = False
    df_comp["comp"] = True
    df = pd.concat([df_train, df_test, df_comp], ignore_index=True)

    logger.info("Removing duplicates (some in the training and competition sets)")
    df = df[(~df.duplicated()) | (df.comp)]

    logger.info(
        "Dropping unnecessary columns not available in the validation set (all null)"
    )
    columns_to_drop = ["USERID", "PHONEID", "TIMESTAMP"]
    df.drop(columns=columns_to_drop, inplace=True)

    logger.info("Making all columns lowercase for easier typing")
    df.columns = [column.lower() for column in df.columns]

    logger.info(
        "Changing missing of WAP columns from 100 to -110 (weaker than all signals)"
    )
    wap_columns = [column for column in df.columns if "wap" in column]
    df[wap_columns] = df[wap_columns].replace(100, -110)

    logger.info("Saving to data/processed")
    df_train = df[df["train"]].copy()
    df_test = df[(~df["train"]) & ~(df["comp"])].copy()
    df_comp = df[df["comp"]].copy()
    df_train.drop(columns=["comp"], inplace=True)
    df_test.drop(columns=["comp"], inplace=True)
    df_comp.drop(columns=["comp", "train"], inplace=True)
    df_train.to_csv(os.path.join("data", "processed", "train.csv"), index=False)
    df_test.to_csv(os.path.join("data", "processed", "test.csv"), index=False)
    df_comp.to_csv(os.path.join("data", "processed", "comp.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
