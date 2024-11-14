from ultralytics import YOLO
import cv2
import torch

class YoloDectector():
    def __init__(self, model_path="yolo11n-pose.pt") -> None:
        self.model = self.load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        
    def load_model(self, model_path):
        model_yolo = YOLO(model_path)  # Load the pose estimation model
        return model_yolo
    
    def predict(self, img):
        results = self.model(img)
        return results
    
    def annotate(self, img, results):
        annotated_frame = results[0].plot(boxes=False)
        return img, annotated_frame
    
    def show_annotations(self, img, annotated_frame):
        cv2.imshow("img",annotated_frame)
        cv2.waitKey(0)
        

