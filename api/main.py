from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List
from api.services.recommendation_service import get_similar_phones, get_top_recommendations, prepare_user_data

app = FastAPI()

class PurchasedProduct(BaseModel):
    product: str

class Rating(BaseModel):
    product: str
    rating: float = Field(..., ge=1, le=5)  # Ensuring rating is a float between 1 and 5
    review: str = ""

class UserActivity(BaseModel):
    user: str
    purchasedProducts: List[PurchasedProduct] = []
    ratings: List[Rating] = []
    viewedProducts: List[str] = []

    class Config:
        json_encoders = {
            float: lambda v: float(v)  # Ensure ratings are treated as floats
        }

@app.get("/")
async def hello():
    return {"message": "Hello, Python Backend!"}

@app.get("/similar_phones/")
async def similar_phones(phone_model: str, top_n: int = 4):
    result = get_similar_phones(phone_model, top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/recommendations/")
async def recommendations(user_activity: UserActivity,top_n: int = Query(default=4)):
    """
    Endpoint to get collaborative filtering recommendations.
    """
    # Prepare user data
    df = prepare_user_data(user_activity.dict())

    # Get top recommendations
    recommendations = get_top_recommendations(user_activity.user, df, top_n)
    return {"user_id": user_activity.user, "recommendations": recommendations}

