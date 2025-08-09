from pydantic import BaseModel, validator
from typing import List

class PredictionRequest(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Features must contain exactly 4 values for Iris dataset')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    predicted_class: str
    confidence: float = None
    timestamp: str