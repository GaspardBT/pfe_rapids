import sys
import pickle

from flask import Flask, request, jsonify

# Your API definition
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if model:
        # use model to predict
        model.predict("ee")

    else:
        return ("No model found")


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345

    model = pickle.load(open("models/model.pkl", "rb"))

    app.run(port=port, debug=True)
