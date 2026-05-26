from typing import Dict, Optional

from fastapi import FastAPI
app = FastAPI()

from pydantic import BaseModel, Field

from app_service.model_service import predict_from_engineered_features

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
app = FastAPI(title="Crypto Volatility Predictor", version="1.0.0")


class PredictRequest(BaseModel):
    volatility_7: float = Field(
        ..., description="Rolling volatility over 7 days of log returns"
    )
    ma_7: float = Field(..., description="7-day moving average of close")
    ma_14: float = Field(..., description="14-day moving average of close")
    hl_range: float = Field(..., description="(high - low) / close")
    volume_mcap_ratio: float = Field(..., description="volume / market_cap")
    dataset_path: Optional[str] = Field(
        None, description="Optional path to dataset.csv"
    )


class PredictResponse(BaseModel):
    predicted_volatility_14: float
    model: str = "xgboost"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    dataset_path = req.dataset_path or "dataset.csv"
    engineered: Dict[str, float] = {
        "volatility_7": req.volatility_7,
        "ma_7": req.ma_7,
        "ma_14": req.ma_14,
        "hl_range": req.hl_range,
        "volume_mcap_ratio": req.volume_mcap_ratio,
    }
    pred = predict_from_engineered_features(
        engineered, dataset_path=dataset_path
    )
    return PredictResponse(predicted_volatility_14=pred)

