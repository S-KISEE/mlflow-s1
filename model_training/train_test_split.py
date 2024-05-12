from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_set(df: pd.DataFrame) -> Tuple:
    """
    Get training and testing sets.

    parameters:
    ------
        df: Dataframe
            Dataframe to be splitted

    return:
    ------
        Training, testing and validation dataframes.
    """
    # train test set
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], test_size=0.4, random_state=42
    )

    # test validation set
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )

    return x_train, x_test, x_val, y_train, y_test, y_val
