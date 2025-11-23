# Emotion Detection Web App  
A real-time emotion recognition system built using **Streamlit**, **OpenCV**, **XGBoost**, **SMOTEENN**, and **PCA**.  
The app captures live video from your webcam and predicts your facial emotion every few seconds using a trained ML model.

## 📌 Features

- 🎥 **Real-time webcam emotion detection**  
- 🧠 **Machine learning pipeline** with MinMaxScaler → PCA (400 components) → XGBoost  
- ⚖️ **Class imbalance handled** using SMOTEENN  
- 📊 **Label encoding** and transformation pipeline saved for inference  
- 🗂️ Fully packaged model files: `model.pkl`, `pipeline.pkl`, `labels.pkl`  
- 🌐 Frontend built with **Streamlit + streamlit-webrtc**

## 📁 Project Structure

```
├── Model_Building.ipynb
├── web-app.py
├── model.pkl
├── pipline.pkl
├── labels.pkl
├── haarcascade_frontalface_default.xml
└── README.md
```

## 🧠 Model Building Details

### ✔️ Preprocessing  
- Convert features to `float32`  
- Train/test split  
- **SMOTEENN** applied  
- MinMax scaling  
- **PCA (n_components=400)**  

### ✔️ Label Encoding  
Stored in `labels.pkl`.

### ✔️ Model  
Trained using **XGBoostClassifier**.

### ✔️ Saved Files  
- `model.pkl`  
- `pipeline.pkl`  
- `labels.pkl`

## 🖥️ Running the Web App

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Run App
```
streamlit run web-app.py
```

## 🎬 How it Works

1. Webcam feed captured  
2. Face detection using Haarcascade  
3. Resize → grayscale → flatten  
4. Pass through preprocessing pipeline  
5. XGBoost predicts emotion  
6. Emotion overlaid on video stream  

## 🚀 Future Improvements
- CNN-based deep models  
- Multi-face detection  
- Cloud deployment  

## 🤝 Contributions
Open to issues & PRs.
