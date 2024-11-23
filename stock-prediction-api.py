from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Optional
import yfinance as yf
from models.predictor import StockPredictor
from data.processor import DataProcessor
from services.sentiment_analyzer import SentimentAnalyzer

app = FastAPI(
    title="StockSage API",
    description="Advanced stock prediction system API",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    symbol: str
    horizon: str  # e.g., "5d", "1mo"
    include_sentiment: bool = False

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    dates: List[str]
    confidence: float
    technical_indicators: dict
    sentiment_score: Optional[float] = None

@app.on_event("startup")
async def startup_event():
    global predictor, data_processor, sentiment_analyzer
    predictor = StockPredictor()
    data_processor = DataProcessor()
    sentiment_analyzer = SentimentAnalyzer()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/symbols")
async def get_available_symbols():
    # Returns list of supported stock symbols
    return {"symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]}

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    try:
        # Fetch historical data
        stock = yf.Ticker(request.symbol)
        hist_data = stock.history(period="1y")
        
        if hist_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Process data and generate features
        processed_data = data_processor.process(hist_data)
        
        # Get sentiment score if requested
        sentiment_score = None
        if request.include_sentiment:
            sentiment_score = await sentiment_analyzer.analyze(request.symbol)
        
        # Generate predictions
        predictions, confidence = predictor.predict(
            processed_data, 
            horizon=request.horizon,
            sentiment_score=sentiment_score
        )
        
        # Generate future dates
        last_date = hist_data.index[-1]
        horizon_days = int(request.horizon[:-1])
        future_dates = [
            (last_date + timedelta(days=x)).strftime("%Y-%m-%d")
            for x in range(1, horizon_days + 1)
        ]
        
        # Calculate technical indicators
        technical_indicators = {
            "rsi": float(processed_data["RSI"].iloc[-1]),
            "macd": float(processed_data["MACD"].iloc[-1]),
            "bollinger_band_upper": float(processed_data["BB_upper"].iloc[-1]),
            "bollinger_band_lower": float(processed_data["BB_lower"].iloc[-1])
        }
        
        return PredictionResponse(
            symbol=request.symbol,
            predictions=predictions.tolist(),
            dates=future_dates,
            confidence=float(confidence),
            technical_indicators=technical_indicators,
            sentiment_score=sentiment_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str):
    try:
        stock = yf.Ticker(symbol)
        hist_data = stock.history(period="6mo")
        analysis = data_processor.generate_technical_analysis(hist_data)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)