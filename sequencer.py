# Description: This script contains the function extract_timestamps_scenes_and_objects that reads a file with timestamps and detected objects and extracts the timestamps, scenes and objects with their confidence levels.

# Function to extract timestamps, scenes and objects from a file
# Input : file_path : the path to the file containing the timestamps and detected objects
#         threshold : the threshold in seconds to consider a new scene (default is 30)
# Output : timestamps : a list of timestamps
#          scenes : a list of scenes with start and end timestamps
#          scene_objects : a list of lists of objects detected in each scene with their confidence levels
def extract_timestamps_scenes_and_objects(file_path, threshold=30):
    timestamps = []
    scenes = []
    scene_objects = []

    with open(file_path, 'r') as f:
        scene_start = None
        last_timestamp = None
        current_scene_objects = []
        for line in f:
            if line.startswith('Timestamp:'):
                timestamp = float(line.split(':')[1].split(' ')[1].split(' ')[0])
                timestamp = int(timestamp)
                timestamps.append(timestamp)

                if scene_start is None:
                    scene_start = timestamp
                elif last_timestamp is not None and timestamp - last_timestamp > threshold:
                    scenes.append([scene_start, timestamps[-2]])
                    scene_start = timestamp
                    scene_objects.append(current_scene_objects)
                    current_scene_objects = []

                detected_objects = line.split(', ')[2].split(': ')[1]
                confidence = float(line.split(', ')[3].split(': ')[1])
                for obj in detected_objects.split(', '):
                    if obj not in [o[0] for o in current_scene_objects]:
                        current_scene_objects.append((obj, confidence))
                
                last_timestamp = timestamp

        scenes.append([scene_start, timestamps[-1]])
        scene_objects.append(current_scene_objects)

    return timestamps, scenes, scene_objects

# Test the function with the file alcohol_detections.txt
file_path = 'alcohol_detections.txt'
all_timestamps, scenes, scene_objects = extract_timestamps_scenes_and_objects(file_path)
print("All timestamps:", all_timestamps)
print("Scenes:", scenes)
print ("Scene objects:", scene_objects)
for i, scene in enumerate(scene_objects):
    print(f"Scene {i+1} objects and confidence:")
    for obj, confidence in scene:
        print(f"- {obj}: {confidence}")
