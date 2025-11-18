import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import pickle as pkl
import time

# ------------------------
# Load your emotion model
# ------------------------
@st.cache_resource
def load_model():
    model = pkl.load(open('model.pkl','rb'))
    pipline = pkl.load(open('pipline.pkl','rb'))
    labels = pkl.load(open('labels.pkl', 'rb'))
    return model, pipline, labels

model, pipline, labels = load_model()


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    if len(face_rect) == 0:
        return None

    x, y, w, h = face_rect[0]
    cropped_image = face_img[y:y+h, x:x+w]
    return cropped_image

# ------------------------
# Video processor class
# ------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_predict_time = 0
        self.current_emotion = "Detecting..."
    
    def recv(self, frame):
        # frame.resize((48, 48)).convert('L')
        img = frame.to_ndarray(format="bgr24")

        # Run prediction every 5 seconds
        if time.time() - self.last_predict_time >= 2:
            self.last_predict_time = time.time()

            try:
                cropped_face = adjusted_detect_face(img)
                if cropped_face is None:
                    self.current_emotion = "No Face Detected"
                else:
                    # ---- Preprocess image as per your model needs ----
                    gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (48, 48))
                    processed = resized.reshape(1, -1)
                    X = pipline.transform(processed)
                    # print(processed.shape)

                    emotion = model.predict(X)
                    self.current_emotion = labels.inverse_transform(emotion)
                    # self.current_emotion = emotion
            except Exception as e:
                self.current_emotion = f"Error: {e}"
                print(e)

        # Add text overlay on video
        cv2.putText(
            img,
            f"Emotion: {self.current_emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------
# Streamlit UI
# ------------------------
st.title("Automatic Emotion Detection Every 5 Seconds")

webrtc_streamer(
    key="emotion-stream",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)