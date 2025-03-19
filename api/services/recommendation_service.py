import numpy as np
from fuzzywuzzy import process
from api.utils.fuzzy_utils import find_closest_match
from pathlib import Path
import joblib
import pandas as pd

# Load the similarity matrix and scaler
script_dir = Path(__file__).resolve().parent.parent
similarity_df = joblib.load(script_dir / '../ML_Model/similarity_matrix.pkl')
scaler = joblib.load(script_dir / '../ML_Model/scaler.pkl')

# Load Collaborative Filtering Model
recommend_model = joblib.load(script_dir / '../ML_Model/trained_cf_model.pkl')

# -------------------- CONTENT-BASED FILTERING --------------------
def get_similar_phones(phone_model, top_n=4):
    """
    Finds similar phones based on content-based filtering.
    """
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

# -------------------- COLLABORATIVE FILTERING --------------------
def get_top_recommendations(user_id, df, top_n=4):
    """
    Generates product recommendations using Collaborative Filtering.
    """
    unique_products = df["product"].unique()  # Get all unique product IDs

    # Predict ratings for all products for this user
    predictions = [recommend_model.predict(user_id, pid) for pid in unique_products]
    predictions.sort(key=lambda x: x.est, reverse=True)  # Sort by estimated rating

    recommended_products = [pred.iid for pred in predictions[:top_n]]
    return recommended_products

def prepare_user_data(user_activity):
    """
    Prepares user interaction data for collaborative filtering.
    """
    interaction_data = []
    user_id = user_activity["user"]

    # Track purchases and ratings
    purchased_products = {p["product"]: True for p in user_activity["purchasedProducts"]}
    rated_products = {r["product"]: r["rating"] for r in user_activity["ratings"]}

    # Add viewed products with neutral rating
    for product in user_activity["viewedProducts"]:
        interaction_data.append({
            "user": user_id,
            "product": product,
            "rating": 2.5  # Neutral weight for views
        })

    # Add purchases with a higher rating (if not rated already)
    for product in purchased_products:
        if product in rated_products:
            interaction_data.append({
                "user": user_id,
                "product": product,
                "rating": rated_products[product]
            })
        else:
            interaction_data.append({
                "user": user_id,
                "product": product,
                "rating": 3.5  # Assign a default score for purchases
            })

    return pd.DataFrame(interaction_data)