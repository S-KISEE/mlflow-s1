import json
import os

import flask
import mlflow
import pandas as pd
from flask import Response, request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# host and port for running the server
HOST = "0.0.0.0"
PORT = 8000

# model name and its stage used in model registry
model_name = "rf_classifier"
version = 1

# required columns needed from the client side to predict
columns_required = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


# before requesting any request loading the model
with app.app_context():
    global model

    # name of the model in model registry
    model_name = os.getenv("MODEL_NAME")

    # configuring the model uri
    model_uri = f"models:/{model_name}/{version}"

    # Load the MLflow model using mlflow.pyfunc.load_model()
    print("---------Loading model--------")
    model = mlflow.sklearn.load_model(model_uri)
    print("--------Model Load Successful--------")


# test route
@app.route("/", methods=["GET"])
def home():
    return "<h1>Flask Server Running</h1>"


@app.route("/predict", methods=["GET"])
def predict():
    # get request
    if request.method == "GET":
        # data to store the data from the client side
        data = {}

        # configuring the data into dictionary to convert it to df
        for column in columns_required:
            data[column] = [request.args.get(column)]

        # converting to dataframe
        data_df = pd.DataFrame(data)

        # predicting the result
        result = model.predict(data_df)

        # returning the result
        return Response(
            status=200,
            content_type="application/json",
            response=json.dumps({"result": str(result[-1])}),
        )

    return Response(
        status=404,
        content_type="application/json",
        response=json.dumps({"message": "Not Found"}),
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
