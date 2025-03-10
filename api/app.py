import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fuzzywuzzy import process
import uvicorn

# Load the similarity matrix and scaler
similarity_df = joblib.load('ML_Model/similarity_matrix.pkl')
scaler = joblib.load('ML_Model/scaler.pkl')

app = FastAPI()

# Function to find the closest match
def find_closest_match(phone_model):
    match = process.extractOne(phone_model, similarity_df.index)
    if match is None:
        return None
    best_match, score = match
    return best_match if score > 75 else None

# Function to get similar phones
def get_similar_phones(phone_model, top_n=4):
    if phone_model not in similarity_df.index:
        closest_match = find_closest_match(phone_model)
        if closest_match:
            phone_model = closest_match
        else:
            return {"error": f"No similar phones found for '{phone_model}'"}

    # Get similarity scores
    similarity_scores = similarity_df.loc[phone_model].values

    # Get top N similar phones (excluding the phone itself)
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]  # Sort in descending order
    similar_phones = similarity_df.iloc[similar_indices].index.tolist()  # Get phone model names

    return {"model": phone_model, "similar_phones": similar_phones}

# API endpoints 
@app.get("/")
async def hello():
    return {"message": "Hello, Python Backend!"}

# API endpoint to get similar phones
@app.get("/similar_phones/")
async def similar_phones(phone_model: str, top_n: int = 4):
    result = get_similar_phones(phone_model, top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)