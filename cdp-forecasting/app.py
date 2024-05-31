import os
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, abort, jsonify
from functions import time_series_prediction
import logging
import sys

# Configure the logging level and format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Initialize the Flask app
flask_app = Flask(__name__)

books = []

@flask_app.route("/")
def root():
    message = { "message": "Flask Prediction!", "version":"0.0.1" }
    return message , 201
    
@flask_app.route('/book', methods=['POST', 'GET'])
def book():
  if request.method == 'POST':
    body = request.get_json()
    books.append(body)
    
    return { "message": "Book already add to database", "body": body }, 201
  elif request.method == 'GET':
    return { "books": books }, 200

@flask_app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    result = time_series_prediction(body)
    print(result)
    # Convert the DataFrame to a dictionary
    # result_dict = result.to_dict(orient='records')  # or use 'index' or 'columns' depending on your preference

    result_dict ={ 
        "XAxis": result.index.strftime('%Y-%m-%d').tolist(),
        "YAxis": result['YAxis'].fillna(0).tolist(),
        "ForecastedCount": result['ForecastedCount'].fillna(0).tolist()
        }
    
    return jsonify(result_dict), 200

# Run the Flask app
if __name__ == "__main__":
    logging.info("Flask app started")
    flask_app.run(host="0.0.0.0", port=8000)