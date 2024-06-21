import os 

from utils.video_editor import get_video_duration,extract_image
from agents.image_analysis_agent import ImageAnalysisAgent
from factories.ollama_llm_factory import OllamaLlmFactory


llm_factory = OllamaLlmFactory(model_analyst="llama3:instruct", model_image_analyst="llava:13b")

image_analysis_llm = llm_factory.create_llm_image_analyst()

image_analysis_agent = ImageAnalysisAgent(model=image_analysis_llm)

video_path = "le_diner_de_con.mp4"

video_duration = get_video_duration(video_path)

print("video duration : " + str(video_duration))

i = 0 

while i < video_duration:
    
        image_path = extract_image(video_path, i)

        print(image_analysis_agent.analyze(image_path = image_path, objects_description = ""))

        # Delete the image file
        os.remove(image_path)
        i += 1


