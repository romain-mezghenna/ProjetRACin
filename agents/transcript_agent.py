from agents.agent import Agent
from models.transcript_model import Transcripter, TranscriptionResult

# Class for the transcript agent
class TranscriptAgent(Agent):
    def __init__(self,model : Transcripter):
        super().__init__(model)
        pass
    # Transcribe the video
    # video_path: str - Path to the video file
    # Returns the transcription
    def analyze(self,video_path : str) -> TranscriptionResult:
        return self.model.transcribe(video_path=video_path)
    

