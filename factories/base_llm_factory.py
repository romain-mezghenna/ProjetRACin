from langchain_core.language_models.llms import BaseLLM
# LlmFactory is an abstract class that defines the interface for creating LLMs models
class BaseLlmFactory():
    def __init__(self, model_analyst: str, model_image_analyst: str):
        self.model_analyst = model_analyst
        self.model_image_analyst = model_image_analyst
        pass
    
    def create_llm_analyst(self) -> BaseLLM:
        pass

    def create_llm_image_analyst(self) -> BaseLLM:
        pass

    def create_llm_formatter(self) -> BaseLLM:
        pass


    
    