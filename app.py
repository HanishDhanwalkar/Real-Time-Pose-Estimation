import cv2
import streamlit as st
import time
from yolo_pose_estimation import YoloDectector

if __name__ == '__main__':
    
    yolo = YoloDectector(model_path="yolo11n-pose.pt") 
    
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        
        # _, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame)
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = yolo.predict(frame)
        frame, annotated_frame = yolo.annotate(frame, results)    
        
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)
        
    else:
        cap.release()
        cv2.destroyAllWindows()
        st.write('Stopped')

    