from ultralytics import YOLO
import time
import cv2
import numpy as np

class DetectorConfig:
    model_path: str = "models/yolov8m.pt"
    person_id: int = 0
    conf_thres: float = 0.25
 
def draw_boxes(img, boxes, color=(0, 255, 0), thickness=2):
    draw_img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
    return draw_img
 
class Detector:
    def __init__(self, config: DetectorConfig = DetectorConfig()):
        model_path = config.model_path
        if not model_path.endswith(".pt"):
            model_path = model_path.split(".")[0] + ".pt"
        self.model = YOLO(model_path)
        self.person_id = config.person_id
        self.conf_thres = config.conf_thres
 
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.detect(img)
 
    def detect(self, img: np.ndarray) -> np.ndarray:
        bboxes_list = []
        for img_idx in range(len(img)):
            img_ = img[img_idx]
            start = time.perf_counter()
            results = self.model(img_, conf=self.conf_thres)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)
            # Filter out only person
            person_detections = detections[detections[:, -1] == self.person_id]
            boxes = person_detections[:, :-2].astype(int)
            # boxes[:,:-1] = boxes[:,:-1].astype(int)
            print(f"Detection inference took: {time.perf_counter() - start:.4f} seconds")

            bboxes_list.append(boxes)
        
        return bboxes_list