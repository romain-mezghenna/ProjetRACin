from factories.object_detector_factory import ObjectDetectorFactory
from models.object_detection_model import ObjectDetector
from models.object_detection_model import ObjectDetectionResult
from models.object_detection_model import Detection
import os
import cv2
from ultralytics import YOLO
import time

class YOLOv8ObjectDetectorFactory(ObjectDetectorFactory):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def create_object_detector(self):
        class Yolov8ObjectDetector(ObjectDetector):

            def __init__(self, model_path: str):
                super().__init__(model_path)
                pass

            def detect_objects(self,video_path : str) -> ObjectDetectionResult:

                output_file = f'./object_detection_results/alcohol_detections_{video_path}.json'
                # If the results file already exists, return the detections
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        data = f.read()
                        return ObjectDetectionResult.model_validate_json(data)
                    
                # Function to detect alcohol objects in a video and save normalized data in JSON format
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                model = YOLO(self.model_path)
                threshold = 0.5
                alcohol_detected_timestamps = []
                alcohol_objects = ["wine glass", "bottle", "cup", "cocktail", "whisky bottle", "beer", "beer glass", "beer bottle"]
                output_results_dir = './object_detection_results'
                output_image_dir = './object_detection_results/images'
                os.makedirs(output_results_dir, exist_ok=True)
                os.makedirs(output_image_dir, exist_ok=True)
                last_detection_time = 0

                    
                # Start time
                start_time = time.time()
                while ret:
                    results = model(frame)[0]

                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = result

                        if score > threshold:
                            object_name = results.names[int(class_id)].lower()
                            if any(obj in object_name for obj in alcohol_objects):
                                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                                if timestamp - last_detection_time >= 1:
                                    image_name = f'detection_{video_path}_{int(timestamp)}.jpg'
                                    image_path = os.path.join(output_image_dir, image_name)
                                    cv2.imwrite(image_path, frame)
                                    alcohol_detected_timestamps.append(Detection(timestamp=timestamp, object_detected=object_name, confidence=score, image_path=image_path))
                                    last_detection_time = timestamp
                        
                    ret, frame = cap.read()
                end_time = time.time()
                cap.release()
                cv2.destroyAllWindows()

                results = ObjectDetectionResult(video_file=video_path, duration=end_time-start_time, model=self.model_path, results=[d.dict() for d in alcohol_detected_timestamps])

                # Save normalized detection data in JSON format
                with open(output_file, 'w') as f:
                    json = results.model_dump_json(indent=2)
                    f.write(json)

                # return the dictionary of detections
                return results
        return Yolov8ObjectDetector(model_path=self.model_name)
                