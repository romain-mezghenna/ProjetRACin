import ollama

# Description: This script contains the function extract_timestamps_scenes_and_objects that reads a file with timestamps and detected objects and extracts the timestamps, scenes and objects with their confidence levels.

# Function to extract timestamps, scenes and objects from a file
# Input : file_path : the path to the file containing the timestamps and detected objects
#         threshold : the threshold in seconds to consider a new scene (default is 30)
# Output : timestamps : a list of timestamps
#          scenes : a list of scenes with start and end timestamps
#          scene_objects : a list of lists of objects detected in each scene with their confidence levels
#          images : a list of lists of images corresponding to each scene    
def extract_detections(file_path, threshold=30):
    timestamps = []
    scenes = []
    scene_objects = []
    images = []

    with open(file_path, 'r') as f:
        scene_start = None
        last_timestamp = None
        current_scene_objects = []
        current_scene_images = []

        for line in f:
            if line.startswith('Timestamp:'):
                timestamp = float(line.split(':')[1].split(' ')[1].split(' ')[0])
                timestamp = int(timestamp)
                timestamps.append(timestamp)
                image_path = line.split(', ')[1].split(': ')[1].strip()
                current_scene_images.append(image_path)
                if scene_start is None:
                    scene_start = timestamp
                elif last_timestamp is not None and timestamp - last_timestamp > threshold:
                    scenes.append([scene_start, timestamps[-2]])
                    scene_start = timestamp
                    scene_objects.append(current_scene_objects)
                    current_scene_objects = []
                    images.append(current_scene_images)
                    current_scene_images = []

                detected_objects = line.split(', ')[2].split(': ')[1]
                confidence = float(line.split(', ')[3].split(': ')[1])
                
                for obj in detected_objects.split(', '):
                    if obj not in [o[0] for o in current_scene_objects]:
                        current_scene_objects.append((obj, confidence))
                
                last_timestamp = timestamp

        scenes.append([scene_start, timestamps[-1]])
        scene_objects.append(current_scene_objects)
        images.append(current_scene_images)

    return timestamps, scenes, scene_objects, images

# Test the function with the file alcohol_detections.txt
file_path = 'alcohol_detections.txt'
all_timestamps, scenes, scene_objects, images = extract_detections(file_path)
print("All timestamps:", all_timestamps)
print("Scenes:", scenes)
print ("Scene objects:", scene_objects)
for i, scene in enumerate(scene_objects):
    print(f"Scene {i+1} objects and confidence:")
    for obj, confidence in scene:
        print(f"- {obj}: {confidence}")

print ("Images:", images)


# response = ollama.generate(
#             model="llava:13b",
#             system="""You're a movie scene analyst and you're task is to describe the scene in the image given. You attention should be turned to the alcohol presence in the image. You're response is expected to be around 2 lines.""",
#             prompt="Describe what is going on in this image, you're response must not exceed 3 lines :",
#             images=["./detections/detection_95.jpg"]
#         )

# print(response.get('response'))