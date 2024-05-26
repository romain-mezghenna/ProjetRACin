from agents.agent import Agent
from models.object_detection_model import ObjectDetector, ObjectDetectionResult

# Class for the transcript agent
class ObjectDetectionAgent(Agent):
    def __init__(self,model : ObjectDetector):
        super().__init__(model)
        pass
    # Detect alcohol objects in the video
    # video_path: str - Path to the video file
    # model_path: str - Path to the model file (Just the name and the library will find the model and download it)
    # Returns the detection result

    def analyze(self,video_path : str) -> ObjectDetectionResult:
        return self.model.detect_objects(video_path=video_path)
    

