from pydantic import BaseModel
from typing import List

class SceneObject(BaseModel):
    object_name: str
    # confidence: float
    count: int

    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['confidence'] = round(result['confidence'], 2)
        return result
    

# Pydantic model for representing a scene
class Scene(BaseModel):
    start_time: float
    end_time: float
    objects_detected: List[SceneObject]
    image_paths: List[str]
    transcript : str = ""
    image_analysis : List[str] = []
    scene_analysis : str = ""
    analysis_duration : float = 0.0


    def dict(self, **kwargs):
        result = super().dict(**kwargs)
        result['start_time'] = round(result['start_time'], 2)
        result['end_time'] = round(result['end_time'], 2)
        result['transcript'] = bytes(result['transcript'], "utf-8").decode("unicode_escape")
        return result
    
    def get_context(self):
        objs = []
        for obj in self.objects_detected:
            objs.append({"object_name": obj.object_name, "count": obj.count})
        return {
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time,2),
            "objects_detected": objs,
            "transcript": self.transcript,
            "image_analysis": self.image_analysis
        }
    def add_image_analysis(self, analysis : str):
        self.image_analysis.append(analysis)

    def set_transcript(self, transcript : str):
        self.transcript = transcript

    def set_scene_analysis(self, analysis : str):
        self.scene_analysis = analysis
    
    def set_analysis_duration(self, duration : float):
        self.analysis_duration = duration
    
        
