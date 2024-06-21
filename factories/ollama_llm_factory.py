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
    
    def create_llm_grader(self):
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
            SYSTEM You are a filtering agent. Your task is to filter and return only the relevant documents retrieved by a vector database in the context of the query. Format the response as JSON: {"relevant_documents": [list of str]}.
            TEMPLATE """{{ if .System }}system
            {{ .System }}
            {{ end }}{{ if .Prompt }}user
            {{ .Prompt }}
            {{ end }}assistant
            {{ .Response }}"""

            MESSAGE system Query: What is the capital of France? Documents: [Document 1: "Paris is the capital of France.", Document 2: "France has many beautiful cities including Paris, Lyon, and Marseille.", Document 3: "The Eiffel Tower is located in Paris."]
            MESSAGE assistant {"relevant_documents": ["Paris is the capital of France."]}

            MESSAGE system Query: Explain the process of photosynthesis. Documents: [Document 1: "Photosynthesis is a process used by plants.", Document 2: "It involves converting light energy into chemical energy.", Document 3: "Photosynthesis takes place in the chloroplasts."]
            MESSAGE assistant {"relevant_documents": ["Photosynthesis is a process used by plants.", "It involves converting light energy into chemical energy.", "Photosynthesis takes place in the chloroplasts."]}

            MESSAGE system Query: What are the benefits of exercise? Documents: [Document 1: "Exercise can help with weight loss.", Document 2: "Regular exercise improves mental health.", Document 3: "Exercise has many benefits."]
            MESSAGE assistant {"relevant_documents": ["Exercise can help with weight loss.", "Regular exercise improves mental health."]}
        '''
        # Check if the model already exists
        try: 
            ollama.show('grader-agent')
        except:
            # Create the model
            ollama.create(model='grader-agent', modelfile=modelfile)
        
        return Ollama(model="grader-agent")
    