# Real-Time-Pose-Estimation

## Instructions
`git clone https://github.com/HanishDhanwalkar/Real-Time-Pose-Estimation.git` \
`cd Real-Time-Pose-Estimation`

Download YOLO weights from [link](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt)

Start Streamlit web app:
`python -m streamlit run app.py`

This will open up web app directly on you browser. \
(If not automatically opened go to http://localhost:8501)

Click RUN


For pose estimation from video file \
`python main.py --video_path <file path>.mp4`

eg.  `python main.py --video_path ./sample_vids/sample1.mp4`

For pose estimation from camera feed \
`python main.py`

