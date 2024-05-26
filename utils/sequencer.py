from collections import defaultdict
from models.scenes_model import Scene,SceneObject
from typing import List
from models.object_detection_model import Detection
# Description: This script contains the function extract_timestamps_scenes_and_objects that reads a file with timestamps and detected objects and extracts the timestamps, scenes and objects with their confidence levels.
# Example of line in the input file: timestamp: 60.53 s, image_path: ./detections\detection_harry-potter.mp4_60.jpg, object: bottle, score: 0.5637896656990051

# Function to extract timestamps, scenes, and objects from a file
# Input : file_path : the path to the file containing the timestamps and detected objects
#         threshold : the threshold in seconds to consider a new scene (default is 30)
# Output : timestamps : a list of timestamps
#          scenes : a list of scenes with start and end timestamps
#          scene_objects : a list of lists of objects detected in each scene with their confidence levels and detection counts
#          images : a list of lists of images corresponding to each scene    
# Function to extract scenes from a file
def extract_detections(data : List[Detection], threshold=30) -> List[Scene]:
    scenes = []
    current_scene_start = None
    last_timestamp = None
    current_scene_objects = defaultdict(lambda: [0, 0])
    current_scene_images = []
    
    for entry in data:
        timestamp = int(entry.timestamp)
        image_path = entry.image_path
        detected_object = entry.object_detected
        confidence = entry.confidence
        
        # Initialize the start time of the current scene
        if current_scene_start is None:
            current_scene_start = timestamp
        # Check if it's time to start a new scene
        elif last_timestamp is not None and timestamp - last_timestamp > threshold:
            scenes.append(Scene(
                start_time=current_scene_start,
                end_time=last_timestamp,
                objects_detected=[SceneObject(object_name=obj, count=conf[0]) for obj, conf in current_scene_objects.items()],
                image_paths=current_scene_images
            ))
            current_scene_start = timestamp
            current_scene_objects = defaultdict(lambda: [0, 0])
            current_scene_images = []

        # Update scene objects and images
        # current_scene_objects[detected_object][0] += confidence
        current_scene_objects[detected_object][0] += 1
        current_scene_images.append(image_path)

        last_timestamp = timestamp

    # Handle the last scene
    if current_scene_start is not None:
        scenes.append(Scene(
            start_time=current_scene_start,
            end_time=last_timestamp,
            objects_detected=[SceneObject(object_name=obj, count=conf[0]) for obj, conf in current_scene_objects.items()],
            image_paths=current_scene_images
        ))

    return scenes