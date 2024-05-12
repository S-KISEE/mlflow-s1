from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_pipeline(numerical_features: List[str], categorical_features: List[str]) -> Pipeline:
    """
    Get sklearn pipeline.

    parameters:
    ------
    numerical_features: List[str]
        List of numerical features.

    categorical_features: List[str]
        List of categorical features.

    return:
    ------
        Sklearn pipeline.
    """
    # creating a transformation pipeline
    transformer = ColumnTransformer(
        [
            ("numerical_imputer", SimpleImputer(strategy="median"), numerical_features),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # pipeline
    pipeline = Pipeline([("transformer", transformer), ("classifier", RandomForestClassifier())])

    return pipeline
