import os
import cv2
from ultralytics import YOLO

# Function to detect alcohol objects in a video
# Input : video_path : the path to the video file
# Output : True if the detection is successful and the results are saved in "./detections" directory and "alcohol_detections.txt" file
def alcohol_objects_detection(video_path):
    if os.path.exists(f'./alcohol_detections_{video_path}.txt'):
        print(f"Alcohol detection results for video '{video_path}' already exist.")
        return True
        
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    model_path = './yolov8n.pt'
    model = YOLO(model_path)
    threshold = 0.5
    alcohol_detected_timestamps = []
    # Liste des termes généraux pour les objets d'alcool
    alcohol_objects = ["wine glass", "bottle", "cup", "cocktail", "whisky bottle", "beer", "beer glass", "beer bottle"]

    # Répertoire pour sauvegarder les images de détection
    output_dir = './detections'
    os.makedirs(output_dir, exist_ok=True)

    # Fichier texte pour enregistrer les résultats
    output_file = f'./alcohol_detections_{video_path}.txt'

    last_detection_time = 0  # Pour garder en mémoire le dernier timestamp enregistré

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                object_name = results.names[int(class_id)].lower()  # Convertir en minuscules pour la comparaison
                if any(obj in object_name for obj in alcohol_objects):
                    # Convertir le numéro de frame en horodatage
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                    # Vérifier si l'intervalle de temps depuis la dernière détection est supérieur à 1 seconde
                    if timestamp - last_detection_time >= 1:
                        image_name = f'detection_{video_path}_{int(timestamp)}.jpg'
                        image_path = os.path.join(output_dir, image_name)
                        # Affichage de la boîte de détection sur la frame
                        # frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # frame = cv2.putText(frame, object_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        #                     (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imwrite(image_path, frame)
                        alcohol_detected_timestamps.append((timestamp, image_path, object_name, score))
                        last_detection_time = timestamp
          
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    # Enregistrer les timestamps et les objets détectés dans un fichier texte
    with open(output_file, 'w') as f:
        for timestamp, image_path, object_name, score in alcohol_detected_timestamps:
            f.write("timestamp: {:.2f} s, image_path: {}, object: {}, score: {}\n".format(timestamp, image_path, object_name, score))
    return True