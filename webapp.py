from fastapi import FastAPI
import pandas as pd
from annoy import AnnoyIndex


app = FastAPI()


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


@app.get("/{title}")
async def root(title):
    recommendations = get_recommendations_new(title)
    return recommendations
