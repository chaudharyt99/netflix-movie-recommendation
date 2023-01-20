import flask
import time
from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
from annoy import AnnoyIndex


app = Flask(__name__)

PREDICTION_DICT = dict() 

def get_recommendations_new(title):
    netflix = pd.read_csv("netflix_titles.csv")
    filledna = pd.read_csv("baggwords.csv")
    indices = pd.Series(filledna.index, index=filledna['title'])
    title = title.replace(' ', '').lower()
    idx = indices[title]
    print(idx)
    # Get the pairwsie similarity scores of all movies with that movie
    u = AnnoyIndex(68322, 'angular')
    u.load('annoy100.ann')  # super fast, will just mmap the file
    similar = u.get_nns_by_item(idx, 10)
    return netflix['title'].iloc[similar]


@app.route("/", methods=['GET'])
def Home():
    return 'Hellow World'

@app.route("/recommend")
def predict():
    title = request.args.get("title")
    start_time = time.time()
    recommendations = get_recommendations_new(title)
    response = {}
    response["response"] = {
        "Movie Title": str(title),
        "Recommended Movie Title": list(recommendations),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9999", debug=True)
