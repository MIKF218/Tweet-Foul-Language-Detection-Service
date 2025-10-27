"""
Tweet Classifier API
====================
FastAPI service for classifying tweets as foul/offensive or proper.

Endpoints:
- GET  /          : Health check
- POST /predict   : Classify a tweet
- GET  /metrics   : Get model metadata
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pickle
import uvicorn
import re
from typing import Optional
import os
import numpy as np

# ============================================================================
# 1. INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Tweet Classifier API",
    description="Binary classification API for detecting foul/offensive tweets",
    version="1.0.0"
)

# Add CORS middleware (allows requests from browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 2. LOAD MODEL ARTIFACTS AT STARTUP
# ============================================================================

# Global variables for model artifacts
model = None
vectorizer = None
threshold = None
metadata = None

def load_artifacts():
    """Load pickled model artifacts"""
    global model, vectorizer, threshold, metadata

    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")

        # Load vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded successfully")

        # Load threshold
        with open('threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
        print(f"✓ Threshold loaded: {threshold:.4f}")

        # Load metadata (optional)
        if os.path.exists('model_metadata.pkl'):
            with open('model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            print("✓ Metadata loaded successfully")
        else:
            metadata = {"status": "metadata not available"}

    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found - {e}")
        print("Please ensure model.pkl, vectorizer.pkl, and threshold.pkl exist")
        raise
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        raise

# Load artifacts when app starts
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup"""
    load_artifacts()
    print("\n" + "="*70)
    print("🚀 Tweet Classifier API Started")
    print("="*70)
    print(f"Model: {metadata.get('best_model', 'Unknown')}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Features: {metadata.get('features', 'Unknown')}")
    print("="*70 + "\n")


# ============================================================================
# 3. PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ============================================================================

class TweetRequest(BaseModel):
    """Request schema for tweet classification"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Tweet text to classify",
        example="I love this beautiful day!"
    )

    @validator('text')
    def validate_text(cls, v):
        """Validate tweet text"""
        if not v or not v.strip():
            raise ValueError("Tweet text cannot be empty")
        if len(v.strip()) < 1:
            raise ValueError("Tweet text too short")
        return v.strip()


class TweetResponse(BaseModel):
    """Response schema for classification result"""
    text: str = Field(..., description="Original tweet text")
    prediction: str = Field(..., description="Classification result: 'foul' or 'proper'")
    label: int = Field(..., description="Numeric label: 1 (foul) or 0 (proper)")
    confidence: float = Field(..., description="Model confidence score (0-1)")
    threshold: float = Field(..., description="Decision threshold used")

    class Config:
        schema_extra = {
            "example": {
                "text": "I love this beautiful day!",
                "prediction": "proper",
                "label": 0,
                "confidence": 0.95,
                "threshold": 0.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class MetricsResponse(BaseModel):
    """Model metadata response"""
    model_name: str
    threshold: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


# ============================================================================
# 4. TEXT PREPROCESSING
# ============================================================================

def preprocess_tweet(text: str) -> str:
    """
    Preprocess tweet text (same as training)

    Args:
        text: Raw tweet text

    Returns:
        Cleaned tweet text
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Remove RT indicator
    text = re.sub(r'\brt\b', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================================
# 5. PREDICTION FUNCTION
# ============================================================================

def classify_tweet(text: str) -> dict:
    """
    Classify a tweet as foul or proper

    Args:
        text: Tweet text to classify

    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    text_clean = preprocess_tweet(text)

    if not text_clean:
        raise ValueError("Tweet is empty after preprocessing")

    # Transform to TF-IDF
    text_tfidf = vectorizer.transform([text_clean])

    # Get prediction probability
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0][1]  # Probability of class 1 (foul)
    elif hasattr(model, 'decision_function'):
        # For SVM, normalize decision function to [0, 1]
        decision = model.decision_function(text_tfidf)[0]
        # Simple sigmoid transformation
        proba = 1 / (1 + np.exp(-decision))
    else:
        proba = float(model.predict(text_tfidf)[0])

    # Apply threshold
    prediction_label = 1 if proba >= threshold else 0
    prediction_text = "foul" if prediction_label == 1 else "proper"

    return {
        "text": text,
        "prediction": prediction_text,
        "label": int(prediction_label),
        "confidence": float(proba),
        "threshold": float(threshold)
    }


# ============================================================================
# 6. API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint

    Returns basic API information
    """
    return {
        "message": "Tweet Classifier API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Detailed health check

    Verifies that model artifacts are loaded
    """
    is_healthy = all([
        model is not None,
        vectorizer is not None,
        threshold is not None
    ])

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "threshold_loaded": threshold is not None
    }


@app.post("/predict", response_model=TweetResponse, tags=["Prediction"])
async def predict_tweet(request: TweetRequest):
    """
    Classify a tweet as foul/offensive or proper

    Args:
        request: TweetRequest with tweet text

    Returns:
        TweetResponse with classification result

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Validate model is loaded
        if model is None or vectorizer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service unavailable."
            )

        # Classify tweet
        result = classify_tweet(request.text)

        return TweetResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Model Info"])
async def get_metrics():
    """
    Get model metadata and performance metrics

    Returns:
        Model information and metrics
    """
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not available"
        )

    return MetricsResponse(
        model_name=metadata.get('best_model', 'Unknown'),
        threshold=float(threshold) if threshold else 0.5,
        accuracy=metadata.get('accuracy'),
        precision=metadata.get('precision'),
        recall=metadata.get('recall'),
        f1_score=metadata.get('f1_score')
    )


# ============================================================================
# 7. ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return {
        "error": "Invalid input",
        "detail": str(exc)
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


# ============================================================================
# 8. MAIN (FOR LOCAL DEVELOPMENT)
# ============================================================================

if __name__ == "__main__":
    """
    Run the API locally
    
    Usage:
        python app.py
    
    Then visit: http://localhost:8000/docs
    """
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )