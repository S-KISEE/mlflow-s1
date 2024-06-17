from typing import Tuple

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline


def train_model(
    pipeline: Pipeline,
    run_name: str,
    model_name: str,
    artifact_path: str,
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> Tuple[str, Pipeline]:
    """
    Train a model and log it to MLflow.

    parameters:
    ------
    pipeline: (sklearn.pipeline)
        Pipeline to train.

    run_name: (str)
        Name of the run.

    model_name: (str)
        Name of the model that is to be store to artifact

    artifact_path: (str)
        path of an artifact

    x: (pd.DataFrame)
        Input features.

    y: (pd.DataFrame)
        Target variable.

    return:
    ------
        Run ID.
    """
    # signature of the model that is to be stored to mlflow
    signature = infer_signature(x, y)

    # starting to log in mlflow
    with mlflow.start_run(run_name=run_name) as run:
        # fitting the mlflow model pipeline
        pipeline = pipeline.fit(x, y)

        # logging the model in model registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=model_name,
        )

    # returning the run id and pipeline
    return run.info.run_id, pipeline
