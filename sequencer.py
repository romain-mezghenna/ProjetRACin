# Description: This script contains the function extract_timestamps_scenes_and_objects that reads a file with timestamps and detected objects and extracts the timestamps, scenes and objects with their confidence levels.
# Example of line in the input file: timestamp: 60.53 s, image_path: ./detections\detection_harry-potter.mp4_60.jpg, object: bottle, score: 0.5637896656990051

# Function to extract timestamps, scenes, and objects from a file
# Input : file_path : the path to the file containing the timestamps and detected objects
#         threshold : the threshold in seconds to consider a new scene (default is 30)
# Output : timestamps : a list of timestamps
#          scenes : a list of scenes with start and end timestamps
#          scene_objects : a list of lists of objects detected in each scene with their confidence levels and detection counts
#          images : a list of lists of images corresponding to each scene    
def extract_detections(file_path, threshold=30):
    from collections import defaultdict
    
    timestamps = []
    scenes = []
    scene_objects = []
    images = []

    with open(file_path, 'r') as f:
        scene_start = None
        last_timestamp = None
        current_scene_objects = defaultdict(lambda: [0, 0])  # [total confidence, count]
        current_scene_images = []

        for line in f:
            if line.startswith('timestamp:'):
                timestamp = float(line.split(':')[1].split(' ')[1].split(' ')[0])
                timestamp = int(timestamp)
                timestamps.append(timestamp)
                image_path = line.split(', ')[1].split(': ')[1].strip()
                
                if scene_start is None:
                    scene_start = timestamp
                elif last_timestamp is not None and timestamp - last_timestamp > threshold:
                    scenes.append([scene_start, timestamps[-2]])
                    scene_start = timestamp
                    # Calculate average confidence for each object and prepare the final format for scene_objects
                    scene_objects.append([(obj, conf[0]/conf[1], conf[1]) for obj, conf in current_scene_objects.items()])
                    current_scene_objects = defaultdict(lambda: [0, 0])
                    images.append(current_scene_images)
                    current_scene_images = []

                detected_objects = line.split(', ')[2].split(': ')[1]
                confidence = float(line.split(', ')[3].split(': ')[1])
                
                for obj in detected_objects.split(', '):
                    current_scene_objects[obj][0] += confidence
                    current_scene_objects[obj][1] += 1

                current_scene_images.append(image_path)
                last_timestamp = timestamp

        # Handle the last scene
        scenes.append([scene_start, timestamps[-1]])
        scene_objects.append([(obj, conf[0]/conf[1], conf[1]) for obj, conf in current_scene_objects.items()])
        images.append(current_scene_images)

    return timestamps, scenes, scene_objects, images
