# Basic imports
import os
import sys
import time
from pydantic import BaseModel
from typing import List

# Import vector database 
import chromadb

# Import Embedding models 
from sentence_transformers import SentenceTransformer

# Import Embedding functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import custom utils
from utils.sequencer import extract_detections
from utils.video_editor import get_video_duration,extract_image

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
from agents.grader_agent import GraderAgent


# Import models
from models.scenes_model import Scene

# Checks the number of arguments passed to the script : 
if len(sys.argv) < 2:
    print("Usage: python main_rag.py <video_path>")
    sys.exit(1)


# Get the video path from the command line arguments
video_path = sys.argv[1]


# OllamaFactory
from factories.ollama_llm_factory import OllamaLlmFactory
ollama_factory = OllamaLlmFactory(model_analyst="llama3:instruct", model_image_analyst="llava:13b")

# TranscripterFactory
from factories.faster_whisper_transcripter_factory import FasterWhisperTranscripterFactory
transcripter_factory = FasterWhisperTranscripterFactory("small")

# ObjectDetectorFactory
from factories.yolov8_object_detector_factory import YOLOv8ObjectDetectorFactory
object_detector_factory = YOLOv8ObjectDetectorFactory("yolov8n.pt")

# Create the AI Models
llm_analyst = ollama_factory.create_llm_analyst()
llm_image_analyst = ollama_factory.create_llm_analyst()
llm_grader = ollama_factory.create_llm_grader()
transcripter = transcripter_factory.create_transcripter()
object_detector = object_detector_factory.create_object_detector()

# Create the agents
transcript_agent = TranscriptAgent(model=transcripter)
object_detection_agent = ObjectDetectionAgent(model=object_detector)
image_analysis_agent = ImageAnalysisAgent(model=llm_image_analyst)
scene_analysis_agent = SceneAnalysisAgent(model=llm_analyst)
grader_agent = GraderAgent(model=llm_grader)


# Create the client for the chromaDB
client = chromadb.PersistentClient(path="./chromadb")



# Analyze the video
print(f"Analyzing video using RAG '{video_path}'")

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
    # if the scene length is greater than 30 seconds, analyze the scene
    if scene.end_time - scene.start_time >= 30:
        # Create the collection for the script
        script_collection = client.get_or_create_collection(name=f"{video_path}_{scene.start_time}_collection", embedding_function=SentenceTransformerEmbeddingFunction(model_name="manu/sentence_croissant_alpha_v0.4"))
        # Get the transcription between the start and end of the scene
        transcriptions_scene = []
        for segment in transcript.results:
            if (segment.start_time >= scene.start_time and segment.end_time <= scene.end_time):
                transcriptions_scene.append(segment.sentence)
        # Set the transcription to the scene
        scene.set_transcript(transcriptions_scene)
        # Add the transcript scene to the collection
        i = 0
        for sentence in transcriptions_scene:
            script_collection.add(
                documents=[sentence],
                metadatas=[{"type": "transcript", "source" :video_path}],
                ids=[f"id_{video_path}_transcript_{scene.start_time}_{i}"]
            )
            i += 1
        current_time = scene.start_time
        while current_time < scene.end_time:
            image_path = extract_image(video_path, current_time)
            image_analysis_result = image_analysis_agent.analyze(image_path = image_path, objects_description = scene.objects_detected)
            # Add the image analysis result to the collection
            script_collection.add(
                documents=[image_analysis_result],
                metadatas=[{"type": "image analysis", "source" :video_path}],
                ids=[f"id_{video_path}_image_analysis_{scene.start_time}_{current_time}"]
            )
            os.remove(image_path)
            current_time += 1
        # Scene collection filled 
        print("Scene collection filled")
        # Query the collection
        collection_results = script_collection.query(
            query_texts=["query : alcool"],
            n_results=10,
            include=["documents", "metadatas"]
        )
        print(f"Results : {collection_results}")
        # Analyze the scene
        scene_analysis = scene_analysis_agent.analyze(context = collection_results)
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
    documents : dict
    analysis_duration: float

# Create the analysis result
analysis_result = AnalysisResult(
    video_path=video_path,
    llm_provider="ollama",
    llm_analyst="llama3:instruct",
    llm_image_analyst="llaava:13b",
    transcripter_provider="faster_whisper",
    transcripter_model="small",
    object_detector_provider="yolov8",
    object_detector_model_path="yolov8n.pt",
    scenes=scenes,
    documents=dict(collection_results),
    analysis_duration=round(time.time() - start_analysis_time, 2)
)

# Save the analysis result to a JSON file
output_file = f"{video_path}_rag_without_grader.json"
with open(output_file, "w") as f:
    f.write(analysis_result.model_dump_json(indent=2))

