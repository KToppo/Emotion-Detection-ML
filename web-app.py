import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import pandas as pd
import urllib.request
import pickle as pkl
import time

# ------------------------
# Load your emotion model
# ------------------------
@st.cache_resource
def load_model():
    model_clf_1 = pkl.load(open('models/model_1.pkl','rb'))
    model_clf_2 = pkl.load(open('models/model_2.pkl','rb'))
    model_clf_3 = pkl.load(open('models/model_3.pkl','rb'))
    model_boost_1 = pkl.load(open('models/model-boost_1.pkl','rb'))
    model_boost_2 = pkl.load(open('models/model-boost_2.pkl','rb'))
    
    pipline1 = pkl.load(open('models/pipline_1.pkl','rb'))
    labels1 = pkl.load(open('models/labels_1.pkl', 'rb'))
    pipline2 = pkl.load(open('models/pipline_2.pkl','rb'))
    labels2 = pkl.load(open('models/labels_2.pkl', 'rb'))
    pipline3 = pkl.load(open('models/pipline_3.pkl','rb'))
    labels3 = pkl.load(open('models/labels_3.pkl', 'rb'))

    models = {
        model_clf_1: [pipline1,labels1],
        model_clf_2: [pipline2,labels2],
        model_clf_3: [pipline3,labels3],
        model_boost_1: [pipline1,labels1],
        model_boost_2: [pipline2, labels2]
    }
    return models
    # return model, pipline, labels

models = load_model()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def combinepredic(img, models=models):
    pred = []
    for model, pipline in models.items():
        X = pipline[0].transform(img)
        emotion = model.predict(X)
        pred.append(pipline[1].inverse_transform(emotion)[0])
    return pd.Series(pred).mode()[0]

# ------------------------
# Face Detection
# ------------------------
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]


# ------------------------
# Webcam Processor
# ------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_predict_time = 0
        self.current_emotion = "Detecting..."
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if time.time() - self.last_predict_time >= 2:
            self.last_predict_time = time.time()

            try:
                cropped = detect_face(img)

                if cropped is None:
                    self.current_emotion = "No Face Detected"
                else:
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (48, 48))
                    processed = resized.reshape(1, -1)
                    # X = pipline.transform(processed)

                    # emotion = model.predict(X)[0]
                    self.current_emotion = combinepredic(processed)
                    # self.current_emotion = labels.inverse_transform([emotion])[0]

            except Exception as e:
                self.current_emotion = f"Error: {e}"

        cv2.putText(
            img,
            f"Emotion: {self.current_emotion}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")



# ------------------------
# URL to Image
# ------------------------
def url_to_image(url):
    resp = urllib.request.urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(img, cv2.IMREAD_COLOR_RGB)



# ============================================
#            MULTI-PAGE APP (same file)
# ============================================

st.sidebar.title("Emotion Detector")
page = st.sidebar.radio(
    "Select Mode",
    ["Webcam Emotion Detection", "Image URL Emotion Detection"]
)


# ======================
# PAGE 1: Webcam
# ======================
if page == "Webcam Emotion Detection":
    st.title("Real-Time Emotion Detection (Webcam)")
    webrtc_streamer(
        key="emotion-stream",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )


# ======================
# PAGE 2: URL Image
# ======================
elif page == "Image URL Emotion Detection":
    st.title("Emotion Detection from Image URL")

    url = st.text_input("Enter Image URL")

    predict_btn = st.button("Predict Emotion")

    if predict_btn:
        if url.strip() == "":
            st.error("Please enter a valid image URL")
        else:
            try:
                img = url_to_image(url)
                st.image(img, caption="Input Image")

                face = detect_face(img)

                if face is None:
                    st.error("No face detected in the image")
                else:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (48, 48))

                    processed = resized.reshape(1, -1)
                    # X = pipline.transform(processed)

                    # pred = model.predict(X)[0]
                    emotion = combinepredic(processed)
                    # emotion = labels.inverse_transform([pred])[0]

                    st.success(f"Predicted Emotion: **{emotion}**")

            except Exception as e:
                st.error(f"Error: {e}")