from yolo_pose_estimation import YoloDectector
import cv2
import torch
import os
import time

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_path', type=str, help='Add Video path')
    
    args = parser.parse_args()
    
    yolo = YoloDectector(model_path="yolo11n-pose.pt") 

    # PUT empty path for camera feed
    # video_path = r"./sample_vids/sample1.mp4" # For pose estimation of a video. 
    video_path = args.video_path  # For pose estimation of a video. 
    
    if video_path != None:
        if os.path.exists(video_path):
            print('FOUND video file')
            cap = cv2.VideoCapture(video_path) 
    else:
        print('NOT FOUND video file: ', video_path)
        cap = cv2.VideoCapture(0)  
        
    # SET height and width of video
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        start = time.perf_counter()
        
        if not ret:
            break
        
        results = yolo.predict(frame)
        frame, annotated_frame = yolo.annotate(frame, results)    

        end = time.perf_counter()
        fps = 1 / (end - start)
        
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Pose Estimation', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
