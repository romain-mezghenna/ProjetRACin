
from models.transcript_model import Transcripter
# TranscripterFactory is an abstract class that defines the interface for creating Transcripters

class TranscripterFactory():
    def __init__(self, model_name : str):
        self.model_name = model_name
        pass
    # Returns Transcripter

    def create_transcripter(self) -> Transcripter:
        pass