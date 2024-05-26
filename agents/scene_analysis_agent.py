from agents.agent import Agent
from langchain_core.language_models.llms import BaseLLM

# Class for the scene analysis agent
class SceneAnalysisAgent(Agent):
    def __init__(self, model: BaseLLM):
        super().__init__(model)
        pass

    # Analyze the scene
    # context: str - Description of the scene context including transcription, detected objects, and image analyses
    # Returns the response from the model (str)
    def analyze(self, context: str) -> str:
        prompt = f"""Analyze the following scene context composed of the transcription of the scene, detected objects related to alcohol, and descriptions from image analyses: {context}.
                     Do not make any assumptions beyond the provided information.
                     Assess the probability of alcohol presence based on the given context, base your assessment mainly on the number of detected objects by minutes in the scene.
                     Image analyses are the second most important factor, the transcription is the least important.
                     Be clear and concise in your description and analysis.
                     You're response MUST NOT contain more than 150 words.
                     You're response MUST be comprehensible to a general audience.
                      """

        return self.model.invoke(prompt)
