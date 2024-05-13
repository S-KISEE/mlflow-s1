import json

import flask
import mlflow
import pandas as pd
from flask import Response, request

app = flask.Flask(__name__)
app.config["DEBUG"] = False

# host and port for running the server
HOST = "127.0.0.1"
PORT = 5001

# model name and its stage used in model registry
model_name = "rf_classifier"
stage = "latest"

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
    # model path in model registry
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    print("Model Load Successfull")


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
