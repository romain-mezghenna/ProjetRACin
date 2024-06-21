from agents.agent import Agent
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms import Ollama
import base64

import ollama 

# Class for the image analysis agent
class ImageAnalysisAgent(Agent):
    def __init__(self,model : BaseLLM):
        super().__init__(model)

        pass

    def convert_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")
    
    # Analyze the text
    # objects_description: str - Description of the objects in the image
    # image_path: str - Path to the image
    # Returns the response from the model (str)
    def analyze(self,objects_description : str, image_path : str) -> str:
        # img_64 = self.convert_image_to_base64(image_path)
        # prompt=f"""Describe the image in up to three lines.. 
        #             Be brief and clear."""
        # llm_with_image_context = self.model.bind(images=[img_64])
        # return llm_with_image_context.invoke(prompt)
        res = ollama.chat(
            model="image-analysis:latest",
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe the image in up to three lines. Be brief and clear.',
                    'images': [image_path]
                }
            ]
        )

        return res['message']['content']


# Focus on the following detected objects in the image: {objects_description}