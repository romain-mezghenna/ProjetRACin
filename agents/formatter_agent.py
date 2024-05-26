from agents.agent import Agent
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms import Ollama
import json

# Class for the formatter agent
class FormatterAgent(Agent):
    def __init__(self, model: BaseLLM):
        super().__init__(model)
        pass

    # Format the analysis response
    # analysis_response: str - The response from the scene analyst
    # Returns the formatted JSON response (str)
    def analyze(self, analysis_response: str) -> str:
        prompt = f"""Take the following scene analysis response and format it into a JSON object with "probability" as a float rounded to two decimal places and "description" as a string:
                     {analysis_response}
                     Ensure the JSON object is structured as follows: {{ "probability": float, "description": str }}."""

        response = self.model.invoke(prompt)
        
        # Ensuring proper JSON formatting
        try:
            response = response[response.find("{"):response.find("}")+1]
            formatted_response = json.loads(response)
            formatted_response = json.dumps(formatted_response, indent=2)
        except json.JSONDecodeError:
            formatted_response = "Invalid JSON response"

        return formatted_response
