from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
# LlmFactory is an abstract class that defines the interface for creating LLMs
class LlmFactory(ABC):
    def __init__(self, model_analyst: str, model_image_analyst: str):
        self.model_analyst = model_analyst
        self.model_image_analyst = model_image_analyst
        pass
    # Returns a LLM type from langchain_community.llms
    @abstractmethod
    def create_llm_analyst(self):
        pass
    # Returns a LLM type from langchain_community.llms  
    @abstractmethod
    def create_llm_image_analyst(self):
        pass

from langchain_community.llms import Ollama

class OllamaLlmFactory(LlmFactory):
    def create_llm_analyst(self):
        return Ollama(model=self.model_analyst)
        

    def create_llm_image_analyst(self):
        return Ollama(model=self.model_image_analyst)

    
    