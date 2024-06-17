import pickle
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from model_training.data_preprocessing.data_preprocessing import get_feature_dataframe
from model_training.mlflow_utils import set_or_create_experiment
from model_training.preprocessing_pipeline import get_pipeline
from model_training.train import train_model
from model_training.train_test_split import get_train_test_set
from model_training.utils import get_classification_metrics, get_performance_plots

default_args = {"owner": "shreejan", "retries": 5}


def prepare_df():
    df = get_feature_dataframe()

    # storing the dataset to csv
    df.to_csv("dataset.csv", index=False)


def train_test_split():
    df = pd.read_csv("dataset.csv")

    x_train, x_test, x_val, y_train, y_test, y_val = get_train_test_set(df)

    # storing the splitted datas
    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("x_val.npy", x_val)

    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("y_val.npy", y_val)


def train_classifier(ti):
    x_train = np.load("x_train.npy")
    # x_test = np.load("x_test.npy")
    # x_val = np.load("x_val.npy")

    y_train = np.load("y_train.npy")
    # y_test = np.load("y_test.npy")
    # y_val = np.load("y_val.npy")

    # converting to dataframe
    x_train = pd.DataFrame(x_train)

    # selecting the features from the dataset
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    # initializing the pipeline for the random forest classifier
    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    # if the experiement already exists acitvate it, otherwise create and activate it
    experiment_id = set_or_create_experiment(experiment_name="house_pricing_classifier")

    # training the model, and accessing the mlflow experient run_id
    run_id, model = train_model(
        pipeline=pipeline,
        run_name="classifier_training",
        model_name="rf_classifier",
        artifact_path="models",
        x=x_train[features],
        y=y_train,
    )

    # pushing variables to airflow xcom
    ti.xcom_push(key="experiement_id", value=experiment_id)
    ti.xcom_push(key="run_id", value=run_id)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def predict_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    x_test = np.load("x_test.npy")

    y_pred = model.predict(x_test)

    np.save("y_pred.npy", y_pred)


def mlflow_logging(ti):
    run_id = ti.xcom_pull("run_id")

    y_test = np.load("y_test.npy")
    y_pred = np.load("y_pred.npy")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # classification reports
    classification_metrics = get_classification_metrics(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )

    # performance plots
    performance_plots = get_performance_plots(
        y_true=y_test, y_pred=y_pred, prefix="test"
    )

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


with DAG(
    "model_training_dag_v1",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 6, 6),
    catchup=False,
) as dag:
    prepare_df = PythonOperator(task_id="prepare_df", python_callable=prepare_df)

    data_split = PythonOperator(task_id="data_split", python_callable=train_test_split)

    model_training = PythonOperator(
        task_id="model_training", python_callable=train_classifier
    )

    model_prediction = PythonOperator(
        task_id="model_prediction", python_callable=predict_model
    )

    logging_mlflow = PythonOperator(
        task_id="logging_mlflow", python_callable=mlflow_logging
    )


prepare_df >> data_split >> model_training >> model_prediction >> logging_mlflow
