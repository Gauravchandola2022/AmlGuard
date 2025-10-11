"""
FastAPI referential service for AML 360
Provides exchange rates, high-risk countries, and suspicious keywords endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import time
from typing import Dict, Any
import os
from pathlib import Path

app = FastAPI(title="AML 360 Referentials Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache with TTL
cache = {}
CACHE_TTL = 3600  # 1 hour

def load_referentials():
    """Load referential data from JSON file"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "referentials.json"
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load referentials.json: {e}")
        return get_default_referentials()

def get_default_referentials():
    """Default referential data"""
    return {
        "exchange_rates": {
            "base": "USD",
            "rates": {
                "USD": 1.0,
                "EUR": 0.91,
                "GBP": 0.78,
                "INR": 83.2,
                "CNY": 7.10,
                "JPY": 142.5,
                "AED": 3.67,
                "BRL": 5.00
            }
        },
        "high_risk_countries": {
            "Level_1": ["DE", "US", "FR", "GB", "CA"],
            "Level_2": ["AE", "BR", "IN", "ZA", "MX"],
            "Level_3": ["IR", "KP", "SY", "RU", "CU"],
            "scores": {"Level_1": 2, "Level_2": 4, "Level_3": 10}
        },
        "suspicious_keywords": {
            "keywords": [
                "gift", "donation", "offshore", "cash", "urgent", 
                "invoice 999", "crypto", "Hawala", "Shell", "bearer", 
                "sensitive", "Bitcoin", "anonymous", "untraceable"
            ]
        }
    }

def get_cached_data(key: str, fetch_func, ttl: int = CACHE_TTL) -> Any:
    """Generic cache function with TTL"""
    current_time = time.time()
    
    if key in cache:
        data, timestamp = cache[key]
        if current_time - timestamp < ttl:
            return data
    
    # Cache miss or expired
    data = fetch_func()
    cache[key] = (data, current_time)
    return data

# Load referentials on startup
referentials_data = load_referentials()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AML 360 Referentials Service", "version": "1.0.0", "status": "healthy"}

@app.get("/api/v1/exchange-rates")
async def get_exchange_rates():
    """Get current exchange rates"""
    def fetch_rates():
        return referentials_data["exchange_rates"]
    
    return get_cached_data("exchange_rates", fetch_rates)

@app.get("/api/v1/high-risk-countries")
async def get_high_risk_countries():
    """Get high-risk countries with risk levels and scores"""
    def fetch_countries():
        return referentials_data["high_risk_countries"]
    
    return get_cached_data("high_risk_countries", fetch_countries)

@app.get("/api/v1/suspicious-keywords")
async def get_suspicious_keywords():
    """Get list of suspicious keywords for transaction monitoring"""
    def fetch_keywords():
        return referentials_data["suspicious_keywords"]
    
    return get_cached_data("suspicious_keywords", fetch_keywords)

# Legacy endpoints for backward compatibility
@app.get("/api/exchange-rates")
async def get_exchange_rates_legacy():
    return await get_exchange_rates()

@app.get("/api/high-risk-countries")
async def get_high_risk_countries_legacy():
    return await get_high_risk_countries()

@app.get("/api/suspicious-keywords")
async def get_suspicious_keywords_legacy():
    return await get_suspicious_keywords()

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "cache_size": len(cache),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
