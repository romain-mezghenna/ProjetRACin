from pydantic import BaseModel
from typing import List

# Pydantic model for representing detection data
class Detection(BaseModel):
    timestamp: float
    object_detected: str
    confidence: float
    image_path: str

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['timestamp'] = round(result['timestamp'], 2)
        result['confidence'] = round(result['confidence'], 2)
        return result

    

# Pydantic model for the overall detection result
class ObjectDetectionResult(BaseModel):
    video_file: str
    duration: float
    model: str
    results: List[Detection]

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['duration'] = round(result['duration'], 2)
        return result


class ObjectDetector : 
    def __init__(self,model_path : str):
        self.model_path = model_path

    def detect_objects(self,video_path : str) -> ObjectDetectionResult:
        pass