import os

import mlflow
from data_preprocessing.data_preprocessing import get_feature_dataframe
from mlflow_utils import set_or_create_experiment
from preprocessing_pipeline import get_pipeline
from train import train_model
from train_test_split import get_train_test_set
from utils import get_classification_metrics, get_performance_plots

# constants
EXPERIEMENT_NAME = os.getenv("EXPERIEMENT_NAME")
RUN_NAME = os.getenv("RUN_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")


if __name__ == "__main__":
    # fetching the data from sklearn dataset
    df = get_feature_dataframe()

    # train, test and validation set
    x_train, x_test, x_val, y_train, y_test, y_val = get_train_test_set(df)

    # selecting the features from the dataset
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    # initializing the pipeline for the random forest classifier
    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    # if the experiement already exists acitvate it, otherwise create and activate it
    experiment_id = set_or_create_experiment(experiment_name=EXPERIEMENT_NAME)

    # training the model, and accessing the mlflow experient run_id
    run_id, model = train_model(
        pipeline=pipeline,
        run_name=RUN_NAME,
        model_name=MODEL_NAME,
        artifact_path=ARTIFACT_PATH,
        x=x_train[features],
        y=y_train,
    )

    # testing
    # print(x_test)
    y_pred = model.predict(x_test)

    # classification reports
    classification_metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )

    # performance plots
    performance_plots = get_performance_plots(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )

    # registring the model in model registry
    # register_model(model_name=MODEL_NAME, run_id=run_id, artifact_path=ARTIFACT_PATH)

    # logging in the mlflow
    with mlflow.start_run(run_id=run_id):
        # log metrics
        mlflow.log_metrics(classification_metrics)

        # log params
        mlflow.log_params(model[-1].get_params())

        # log tags
        mlflow.set_tags({"type": "random forest classifier"})

        # log description
        mlflow.set_tag(
            "mlflow.note.content",
            """
            This is a Random Forest Classifier for the house pricing dataset.
            The dataset is loaded from sklearn dataset.
            """,
        )

        # log plots
        for plot_name, fig in performance_plots.items():
            mlflow.log_figure(fig, plot_name + ".png")
