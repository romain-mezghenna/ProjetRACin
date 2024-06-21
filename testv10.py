# from factories.object_detector_factory import ObjectDetectorFactory
# from agents.object_detection_agent import ObjectDetectionAgent

# # Create an ObjectDetectorFactory
# object_detector_factory : ObjectDetectorFactory


# from factories.yolov8_object_detector_factory import YOLOv8ObjectDetectorFactory
# object_detector_factory = YOLOv8ObjectDetectorFactory("yolov10n.pt")

# # Create the AI Models
# object_detector = object_detector_factory.create_object_detector()

# # Create the agents
# object_detection_agent = ObjectDetectionAgent(model=object_detector)


# result = object_detection_agent.analyze(video_path="le_diner_de_con.mp4")

# print(result)

from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov10n.pt")

# Perform object detection on an image
results = model("image.jpg")

# Display the results
results[0].show()