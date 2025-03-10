from fastapi import FastAPI, HTTPException
from api.services.recommendation_service import get_similar_phones

app = FastAPI()

@app.get("/")
async def hello():
    return {"message": "Hello, Python Backend!"}

@app.get("/similar_phones/")
async def similar_phones(phone_model: str, top_n: int = 4):
    result = get_similar_phones(phone_model, top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result