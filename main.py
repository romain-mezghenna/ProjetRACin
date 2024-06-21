# Basic imports
import os
import sys
import time
import random
from pydantic import BaseModel
from typing import List

# Import custom utils
from utils.sequencer import extract_detections

# Import factories 
from factories.base_llm_factory import BaseLlmFactory
from factories.transcripter_factory import TranscripterFactory
from factories.object_detector_factory import ObjectDetectorFactory

# Import agents 
from agents.transcript_agent import TranscriptAgent
from agents.object_detection_agent import ObjectDetectionAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.scene_analysis_agent import SceneAnalysisAgent
from agents.formatter_agent import FormatterAgent

# Import models
from models.scenes_model import Scene



### Arguments parsing (usage : python main.py <video_path> <llm_provider>="ollama" <transcripter_provider>="faster_whisper"  <object_detector_provider>="yolov8")

# Checks the number of arguments passed to the script : 
if len(sys.argv) < 2:
    print("Usage: python main.py <video_path>")
    sys.exit(1)

# LLM provider (ollama, openai, etc.)
llm_provider = "ollama"

# LLM Analyst model 
llm_analyst = "llama3:instruct"

# LLM Image Analyst model
llm_image_analyst = "llava:13b"

# Transcripter provider (faster_whisper,openai, etc.)
transcripter_provider = "faster_whisper"

# Transcripter model
transcripter_model = "small"

# Object detector provider (yolov8, etc.)
object_detector_provider = "yolov8"

# Object detector model path
object_detector_model_path = "./yolov8n.pt"



# Get the optionnal arguments
if len(sys.argv) > 2:
    llm_provider = sys.argv[2]
    if len(sys.argv) > 3:
        transcripter_provider = sys.argv[3]
        if len(sys.argv) > 4:
            object_detector_provider = sys.argv[4]

# Get the video path from the command line arguments
video_path = sys.argv[1]

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Video file '{video_path}' does not exist.")
    sys.exit(1)



# Create a LlmFactory
llm_factory : BaseLlmFactory

if llm_provider == "ollama":
    from factories.ollama_llm_factory import OllamaLlmFactory
    llm_factory = OllamaLlmFactory(model_analyst=llm_analyst, model_image_analyst=llm_image_analyst)
else:
    print(f"LLM provider '{llm_provider}' is not available.")
    sys.exit(1)

# Create a TranscripterFactory
transcripter_factory : TranscripterFactory

if transcripter_provider == "faster_whisper":
    from factories.faster_whisper_transcripter_factory import FasterWhisperTranscripterFactory
    transcripter_factory = FasterWhisperTranscripterFactory(transcripter_model)
else:
    print(f"Transcripter provider '{transcripter_provider}' is not available.")
    sys.exit(1)

# Create an ObjectDetectorFactory
object_detector_factory : ObjectDetectorFactory

if object_detector_provider == "yolov8":
    from factories.yolov8_object_detector_factory import YOLOv8ObjectDetectorFactory
    object_detector_factory = YOLOv8ObjectDetectorFactory(object_detector_model_path)
else:
    print(f"Object detector provider '{object_detector_provider}' is not available.")
    sys.exit(1)

# Create the AI Models
transcripter = transcripter_factory.create_transcripter()
object_detector = object_detector_factory.create_object_detector()
image_analysis_llm = llm_factory.create_llm_image_analyst()
scene_analysis_llm = llm_factory.create_llm_analyst()
formatter_llm = llm_factory.create_llm_formatter()

# Create the agents
transcript_agent = TranscriptAgent(model=transcripter)
object_detection_agent = ObjectDetectionAgent(model=object_detector)
image_analysis_agent = ImageAnalysisAgent(model=image_analysis_llm)
scene_analysis_agent = SceneAnalysisAgent(model=scene_analysis_llm)
formatter_agent = FormatterAgent(model=formatter_llm)

# Analyze the video
print(f"Analyzing video '{video_path}'")
# Start time of the analysis
start_analysis_time = time.time()
# Transcribe the video
transcript = transcript_agent.analyze(video_path=video_path)
# Detect alcohol objects in the video
detections = object_detection_agent.analyze(video_path=video_path)
# Parse the detections in scenes 
scenes = extract_detections(detections.results)
# For each scene
for scene in scenes:
    start_scene_time = time.time()
    # Get the transcription between the start and end of the scene
    transcriptions_scene = []
    for segment in transcript.results:
        if (segment.start_time >= scene.start_time and segment.end_time <= scene.end_time):
            transcriptions_scene.append(segment.sentence)
    # Set the transcription to the scene
    scene.set_transcript(transcriptions_scene)
    # Get random images from the scene
    scene_length = scene.end_time - scene.start_time
    # Determine the number of images to select based on the scene length
     # Determine the number of images to select based on the scene length
    num_images = int(max(1, min(6, scene_length // 60)))  # Adjust the divisor to control selection range
    # Sample randomly from the available image paths
    selected_image_paths = random.sample(scene.image_paths,num_images)
    print(f"Selected {selected_image_paths} images from the scene")

    # Analyze the images
    for image_path in selected_image_paths:
        # Analyze the image
        image_analysis = image_analysis_agent.analyze(image_path=image_path,objects_description=scene.objects_detected)
        # Add the image analysis to the scene
        scene.add_image_analysis(image_analysis)
        print(f"Image analysis for '{image_path}' added to the scene")
    # Analyze the scene
    scene_context = scene.get_context()
    print(f"Analyzing scene {scene.start_time}-{scene.end_time}")
    scene_analysis = scene_analysis_agent.analyze(context=scene_context)
    print(f"Scene analysis for {scene.start_time}-{scene.end_time}: {scene_analysis}")
    # Set the scene analysis
    scene.set_scene_analysis(scene_analysis)
    # Set the analysis duration
    end_scene_time = time.time()
    scene.set_analysis_duration(round(end_scene_time - start_scene_time))

class AnalysisResult(BaseModel):
    video_path: str
    llm_provider: str
    llm_analyst: str
    llm_image_analyst: str
    transcripter_provider: str
    transcripter_model: str
    object_detector_provider: str
    object_detector_model_path: str
    scenes: List[Scene]
    analysis_duration: float

# Create the analysis result
analysis_result = AnalysisResult(
    video_path=video_path,
    llm_provider=llm_provider,
    llm_analyst=llm_analyst,
    llm_image_analyst=llm_image_analyst,
    transcripter_provider=transcripter_provider,
    transcripter_model=transcripter_model,
    object_detector_provider=object_detector_provider,
    object_detector_model_path=object_detector_model_path,
    scenes=scenes,
    analysis_duration=round(time.time() - start_analysis_time, 2)
)

# Save the analysis result to a JSON file
output_file = f"{video_path}.json"
with open(output_file, "w") as f:
    f.write(analysis_result.model_dump_json(indent=2))


