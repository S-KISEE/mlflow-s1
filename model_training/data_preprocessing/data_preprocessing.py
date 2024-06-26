import os

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data() -> pd.DataFrame:
    """
    Download the California housing dataset and return as a pandas dataframe.

    return:
    ------
        California housing dataset as a pandas dataframe.
    """
    # Defining the path to download the dataset
    current_directory = os.path.abspath(os.path.dirname(__file__))

    # Fetching the dataa
    data = fetch_california_housing(
        data_home=f"{current_directory}/data/",
        as_frame=True,
        download_if_missing=True,
    )

    # returning the dataset in the form of dataframe
    return data.frame


def get_feature_dataframe() -> pd.DataFrame:
    """
    Get the feature dataframe.

    return:
    ------
        Feature dataframe.
    """
    # loading the data
    df = load_data()

    # seperating the taget variables
    df["id"] = df.index
    df["target"] = df["MedHouseVal"] >= df["MedHouseVal"].median()
    df["target"] = df["target"].astype(int)

    return df
