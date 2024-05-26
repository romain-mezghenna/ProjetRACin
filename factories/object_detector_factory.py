
from models.object_detection_model import ObjectDetector
# TranscripterFactory is an abstract class that defines the interface for creating Transcripters

class ObjectDetectorFactory():
    def __init__(self, model_name : str):
        self.model_name = model_name
        pass
    # Returns ObjectDetector
    def create_object_detector(self) -> ObjectDetector:
        pass