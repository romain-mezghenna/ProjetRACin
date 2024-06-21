from agents.agent import Agent
from langchain_core.language_models.llms import BaseLLM

# Class for the scene analysis agent
class GraderAgent(Agent):
    def __init__(self, model: BaseLLM):
        super().__init__(model)
        pass

    # Analyze the scene
    # context: str - Description of the scene context including transcription, detected objects, and image analyses
    # Returns the response from the model (str)
    def analyze(self, query: str, documents : str) -> str:
        prompt = f"""Query : {query}
                        Documents : {documents}
                        Filter the documents based on the query and return the relevant documents.
                      """

        return self.model.invoke(prompt)
