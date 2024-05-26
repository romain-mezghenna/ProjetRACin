from factories.base_llm_factory import BaseLlmFactory
from langchain_community.llms import Ollama
import ollama

class OllamaLlmFactory(BaseLlmFactory):
    def create_llm_analyst(self):
        # Checks if the model exists 
        try:
            ollama.show(self.model_analyst)
        except:
            # Pull the model 
            ollama.pull(self.model_analyst)
        
        # Modelfile for the custom model
        modelfile = f"FROM {self.model_analyst}" + '\n' + '''
            PARAMETER temperature 0.7
            PARAMETER num_ctx 8192
            PARAMETER top_k 30
            PARAMETER repeat_penalty 1.2
            PARAMETER top_p 0.7
            SYSTEM You are a scene analyst. Your task is to describe scenes from movies, focusing on interactions between characters and objects, and assess the probability of alcohol presence in the scene.
           
        '''
        #  TEMPLATE """{{ if .System }}system
            # {{ .System }}
            # {{ end }}{{ if .Prompt }}user
            # {{ .Prompt }}
            # {{ end }}assistant
            # {{ .Response }}"""
        # Check if the model already exists
        try: 
            ollama.show('scene-analysis')
        except:
            # Create the model
            ollama.create(model='scene-analysis', modelfile=modelfile)
        
        return Ollama(model="scene-analysis")

        

    def create_llm_image_analyst(self):
        # Checks if the model exists 
        try: 
            ollama.show(self.model_image_analyst)
        except:
            # Pull the model 
            ollama.pull(self.model_image_analyst)
        
        # Modelfile for the custom model
        modelfile = f"FROM {self.model_image_analyst}" + '\n' + '''
            PARAMETER temperature 0.7
            PARAMETER num_ctx 4096
            PARAMETER top_k 30
            SYSTEM You are an image analyst. Your task is to describe scenes in images, focusing on the interaction between detected objects and characters, or noting the lack of interaction if none is present.
            TEMPLATE """{{ if .System }}system
            {{ .System }}
            {{ end }}{{ if .Prompt }}user
            {{ .Prompt }}
            {{ end }}assistant
            {{ .Response }}"""
        '''
        # Check if the model already exists
        try: 
            ollama.show('image-analysis')
        except:
            # Create the model
            ollama.create(model='image-analysis', modelfile=modelfile)
        
        return Ollama(model="image-analysis")
    
    def create_llm_formatter(self):
        # Checks if the model exists 
        try:
            ollama.show(self.model_analyst)
        except:
            # Pull the model 
            ollama.pull(self.model_analyst)
        
        # Modelfile for the custom model
        modelfile = f"FROM {self.model_analyst}" + '\n' + '''
            PARAMETER temperature 0.7
            PARAMETER num_ctx 4096
            PARAMETER top_k 30
            SYSTEM You are a formatter agent. Your task is to format the response from a scene analyst into a JSON format: {"probability": float (rounded to 2 decimal places), "description": str}.
            TEMPLATE """{{ if .System }}system
            {{ .System }}
            {{ end }}{{ if .Prompt }}user
            {{ .Prompt }}
            {{ end }}assistant
            {{ .Response }}"""

            MESSAGE system Analysis of the scene shows multiple objects related to alcohol, such as wine bottles, beer cans, and glasses. Based on the context, the probability of alcohol presence is 80%.
            MESSAGE assistant {"probability": 0.8, "description": "The scene suggests a lively atmosphere with people engaged in conversation. No specific objects related to alcohol are detected."}
        '''
        # Check if the model already exists
        try: 
            ollama.show('formatter-agent')
        except:
            # Create the model
            ollama.create(model='formatter-agent', modelfile=modelfile)
        
        return Ollama(model="formatter-agent")

