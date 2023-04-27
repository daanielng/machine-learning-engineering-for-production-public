import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist

# reference: https://github.com/https-deeplearning-ai/machine-learning-engineering-for-production-public/blob/main/course4/week3-ungraded-labs/C4_W3_Lab_4_Github_Actions/README.md

app = FastAPI(title="Predicting Wine Class with batching")

# Open classifier in global scope
with open("models/wine.pkl", "rb") as file:
    clf = pickle.load(file)


class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}
