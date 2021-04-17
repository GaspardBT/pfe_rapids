import sys
import pickle

from flask import Flask, request
import numpy as np

from utils import load_image, append_to_csv, model_trainer

# Your API definition
app = Flask(__name__)


@app.route("/add-image", methods=["POST"])
def add_image():
    label = request.args.get("label", None)
    if label:
        img = load_image(request)
        img_1d = np.ravel(img)
        append_to_csv("./datastore/imgdata.csv", img_1d)
        return "Image added to database", 200
    return "Label not found", 400


@app.route("/train", methods=["get"])
def train():
    try:
        model_trainer(
            datapath="./datastore/imgdata.csv", modelpath="models/model.pkl", k=3
        )
    except Exception as e:
        return str(e), 500


@app.route("/predict", methods=["POST"])
def predict():
    if model:
        img = load_image(request)
        img_1d = np.ravel(img)
        # use model to predict
        result = model.predict(img_1d)
        return result
    else:
        return "No model found", 500


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345

    model = pickle.load(open("models/model.pkl", "rb"))

    app.run(port=port, debug=True)
