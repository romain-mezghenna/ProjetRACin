import sys
import ollama 
import json
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import videoEditor as ve
from typing import List
from PIL import Image
import base64
from io import BytesIO
import os

mode = "ollama"

import LlmFactory

llm_factory : LlmFactory

if (mode == "ollama"):
    llm_factory = LlmFactory.OllamaLlmFactory(model_analyst="llava:13b", model_image_analyst="llama3:instruct")

# Create the differents LLMs
model_analyst = llm_factory.create_llm_analyst()
model_image_analyst = llm_factory.create_llm_image_analyst()

# Function to get a image analysis from an image file path 
def get_image_analysis(image_path):

    # def convert_to_base64(pil_image: Image):
    #     buffered = BytesIO()
    #     pil_image.save(buffered, format="PNG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    #     return img_str

    # def load_image(image_path: str):
    #     pil_image = Image.open(image_path)
    #     image_b64 = convert_to_base64(pil_image)
    #     print("Loaded image successfully!")
    #     return image_b64
    
    # response = model_image_analyst.invoke("""You're a movie scene analyst and you're task is to describe the scene in the image given. You attention should be turned to the alcohol presence in the image. You're response is expected to be around 2 lines.
    #                   Describe what is going on in this image, you're response must not exceed 3 lines : (This image has alread been detected to have alcohol in it)
    #                   """,images=[load_image(image_path)])
    # return response

    response = ollama.generate(model="llava:13b",
                                system="""You're a movie scene analyst and you're task is to describe the scene in the image given. You attention should be turned to the alcohol presence in the image. You're response is expected to be around 2 lines.""",
                                prompt="""Describe what is going on in this image, you're response must not exceed 3 lines : """,
                                images=[image_path]
                            )
    
    return response.get("response")






# Set the differents Agents

# Scene analyst 
scene_analyst = Agent(
    role="Scene Analyst",
    goal="Analyze the provided scene and determine the probability of alcohol presence, along with a short description of the scene.",
    backstory="""
    As a Scene Analyst Assistant, your task is to analyze the given scene and assess the likelihood of alcohol being present. The user will provide you with the transcript, image analysis, and detected objects with their respective probabilities.
    Your responsibilities include:
    - Providing a concise description of the scene.
    - Be factual and objective in your analysis.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=model_image_analyst
)

transcript_summarizer = Agent(
    role="Transcript Summarizer",
    goal="Summarize the transcript of the scene.",
    backstory="""You're a transcript summarizer, you need to summarize the transcript of the scene.
    You're response must be a summary of the transcript of the scene.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=model_image_analyst
)


alcohol_grader = Agent(
    role="Scene Analyst Assistant",
    goal="Analyze the provided scene description and attribute a probability of alcohol presence.",
    backstory="""
    As a Scene Analyst Assistant, your task is to analyze the description of a scene and determine the likelihood of alcohol being present. 
    The user will provide you with a detailed description of the scene.
    Your responsibilities include:
    - Assessing the probability of alcohol presence in the scene using the following scale: 
      0.2 for not likely, 
      0.5 for maybe (Some objects detected in the scene are related to alcohol but not enough to be certain), 
      0.8 or higher for definite presence.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=model_image_analyst
)

formater = Agent(
    role= "Formater",
    goal="Format the response of the Scene analyst in a JSON format.",
    backstory="""
      You're a formater, you need to format the response of the Scene analyst in a JSON format. 
      You're response must be a JSON formatted string ready to be parsed by the user.
      You're response must have the following keys : 'probability' a float (if it is not a float you need to parse it) and 'description' (a string).
      You're response must not include any other text than the JSON formatted string.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=model_image_analyst
)


# Get a random image from the video video.mp4 at 15 seconds

# image = ve.extract_image("videomp4_out", 13) 

# the path of the image is ./images/video-15.png

def get_scene_analysis(context):
    # Analyze the scene
    task1 = Task(
        agent=scene_analyst,
        description=f"""
        Analyze the scene from the context given and give a precise description of it.
        Context :
        {context} 
        """,
        expected_output="""The description of the scene in less that 4 lines.""",
        tools=[],
    )

    task2 = Task(
        agent=alcohol_grader,
        description=f"""
        Analyze the description of the scene and determine the probability of alcohol presence.
        """,
        expected_output="""A float representing the probability of alcohol presence in the scene. from 0.2 to 0.8.""",
    )

    task3 = Task(
        agent=formater,
        description="""
        Format the response of the Scene analyst in a JSON format.
        """,
        expected_output="""A JSON formatted string ready to be parsed by the user with the keys "probability" and "description".""",
        tools=[],
    )

    crew = Crew(
        agents=[scene_analyst,alcohol_grader,formater],
        tasks=[task1,task2,task3],
        verbose=True,   
    )
    result = crew.kickoff()

    return result


import videoDetector as vd
import transcripter as tr
import sequencer as sq
import randomSelector as rs
import time

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Please provide the path to the video file.")
        return
    video_path = args[0]
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return
    # Create the directory for the outputs
    os.makedirs("outputs", exist_ok=True)
    # Track the time of the process
    print("Starting the process...")
    start_time = time.time()
    # Detect alcohol objects in the video
    success = vd.alcohol_objects_detection(video_path)
    if success:
        print("Alcohol objects detected successfully.")
    else:
        print("Error while detecting alcohol objects in the video.")
        return
    # Do the transcription of the video
    tr.transcribe_audio(model_name="large-v3", video_path=video_path)
    print("Transcription done successfully.")
    # Extract the detections from the file
    file_path = f"alcohol_detections_{video_path}.txt"
    all_timestamps, scenes, scene_objects, images = sq.extract_detections(file_path)
    print("Detections extracted successfully.")
    print("Starting the analysis of the scenes...")
    # Start the analysis of the scenes
    print(scenes)
    for i, scene in enumerate(scenes):
        # Get the context of the scene 
        context = ""
        # Add the length of the scene
        context += f"Scene duration: {scene[1] - scene[0]:.2f} seconds\n"
        #Add the objects detected in the scene
        context += "Scene objects and confidence:\n"

        for obj, confidence, count in scene_objects[i]:
            context += f"- {obj}: Average confidence : {confidence}, Number of detections : {count}\n"
        # Add the transcript in the file : f"transcripts/{os.path.basename(video_path)}-{model_name}.txt"
        transcript_file = f"transcripts/{os.path.basename(video_path)}-large-v3.txt"
        transcript = "Transcript:\n"
        with open(transcript_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                    transcript += line
        summuraize_task = Task(
            agent=transcript_summarizer,
            description=f"""
            Summarize the transcript of the scene.
            {transcript} 
            """,
            expected_output="""The summary of the transcript of the scene.""",
            tools=[],
        )
        crew = Crew(
            agents=[transcript_summarizer],
            tasks=[summuraize_task],
            verbose=True,   
        )
        result = crew.kickoff()
        context += f"Transcript summary: {result}\n"
        # Select random images from the scene to analyze
        print(images)
        images_to_analyze = rs.select_random_images(scene=scene,images=images[i])
        for image in images_to_analyze:
            # Analyze the image
            image_analysis = get_image_analysis(image)
            print(f"Image {image} analyzed successfully.")
            # Add the image analysis to the context
            context += f"Image analysis: {image_analysis}\n"
        # Print the context
        print(f"Scene {i+1} context:\n{context}")
        # Get the scene analysis by the agents 
        result = get_scene_analysis(context)
        
        # extract the data between the {} in the result string for parsing the JSON
        result = result[result.find("{"):result.find("}")+1]
        result = json.loads(result)
        # Add the timestamps of the scene for the user
        result["start_timestamp"] = scene[0]
        result["end_timestamp"] = scene[1]
        result["scene_id"] = i+1
        
        # Save the result in a JSON file
        with open(f"./outputs/{video_path}-scene{i+1}.json", 'w') as f:
            json.dump(result, f)

    print("Analysis of the scenes done successfully.")
    end_time = time.time()
    print(f"Process done in {end_time - start_time:.2f} seconds.")
    
    #Output the analysis details in a JSON file
    with open(f"./outputs/{video_path}-analysis.json", 'w') as f:
        json.dump({"scenes":scenes,"time_completion":f"{end_time - start_time:.2f} seconds","scene_objects":scene_objects,"images":images,"all_timestamps":all_timestamps}, f)
    print("Analysis details saved in the file.")


    

if __name__ == "__main__":
    main()