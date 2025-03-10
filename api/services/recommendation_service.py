import numpy as np
from fuzzywuzzy import process
from api.utils.fuzzy_utils import find_closest_match
from pathlib import Path
import joblib

# Load the similarity matrix and scaler
script_dir = Path(__file__).resolve().parent.parent
similarity_df = joblib.load(script_dir / '../ML_Model/similarity_matrix.pkl')
scaler = joblib.load(script_dir / '../ML_Model/scaler.pkl')

def get_similar_phones(phone_model, top_n=4):
    if phone_model not in similarity_df.index:
        closest_match = find_closest_match(phone_model, similarity_df.index)
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