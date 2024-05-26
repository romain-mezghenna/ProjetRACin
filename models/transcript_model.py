from pydantic import BaseModel
from typing import List

# Define the Segment class
class Segment(BaseModel):
    start_time: float
    end_time: float
    sentence: str

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['start_time'] = round(result['start_time'], 2)
        result['end_time'] = round(result['end_time'], 2)
        result['sentence'] = bytes(result['sentence'], "utf-8").decode("unicode_escape")
        return result
    
# Define the TranscriptionResult class
class TranscriptionResult(BaseModel):
    video_file: str
    duration: float
    model: str
    results: List[Segment]
    language: str
    language_probability: float

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['duration'] = round(result['duration'], 2)
        result['language_probability'] = round(result['language_probability'], 2)
        return result

    

# Define the Transcripter class

class Transcripter:
    def __init__(self, model_name):
        self.model_name = model_name

    def transcribe(self, video_path : str) -> TranscriptionResult:
        pass
