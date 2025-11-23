

# Emotion Detection System Using Ensemble Machine Learning & Streamlit

## 📌 Project Overview
This project is a **Facial Emotion Detection System** built using:
- Machine Learning (XGBoost, SVM, Logistic Regression, Gradient Boosting)
- Ensemble Learning (Stacking + Majority Voting)
- Data Balancing Techniques (SMOTE, SMOTEENN)
- PCA & Scaling Pipelines
- Streamlit Web App (Webcam + URL-based detection)

The goal was to build a **high‑accuracy and balanced model** by experimenting with multiple sampling techniques and classifiers, then combining them to improve overall robustness.

---

## 📂 Folder Structure

```
├── models/
│   ├── model_1.pkl
│   ├── model_2.pkl
│   ├── model_3.pkl
│   ├── model-boost_1.pkl
│   ├── model-boost_2.pkl
│   ├── pipline_1.pkl
│   ├── pipline_2.pkl
│   ├── pipline_3.pkl
│   ├── labels_1.pkl
│   ├── labels_2.pkl
│   ├── labels_3.pkl
│   ├── M1SMOTE_boost.png
│   ├── M1SMOTE_Clf.png
│   ├── M2ENN_boost.png
│   ├── M2ENN_Clf.png
│   ├── M3SMOTE_clf-NE.png
│   └── final_model.png
│
├── image-to-vector.py
├── kaggle_handler.py
├── Model_Building.ipynb
├── Model_Testing.ipynb
├── haarcascade_frontalface_default.xml
├── web-app.py
└── README.md
```

---

## 📊 Dataset Pipeline

### **1. Downloading Dataset**
Using `kaggle_handler.py`, datasets are downloaded and stored in:
```
run
from kaggle_handler import handler
assets = handler('samithsachidanandan/human-face-emotions')
```
It will download data in
```
Assets/Data/<emotion-class>/
```

### **2. Image to CSV Conversion**
`image-to-vector.py`:
- Detects faces using Haar Cascade  
- Crops face region  
- Converts to **48×48 grayscale**  
- Flattens into 2304-d vector  
- Saves batches to **data.csv**

---

## 🧠 Model Building & Experiments (Model_Building.ipynb)

Multiple experiments were performed to improve performance:

---

## **1️⃣ Experiment: SMOTE + class_weight='balanced'**

### XGBoost Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/M1SMOTE_boost.png)

### Stacking Classifier Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/M1SMOTE_Clf.png)

---

## **2️⃣ Experiment: SMOTEENN + class_weight='balanced'**

### XGBoost Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/M2ENN_boost.png)

### Stacking Classifier Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/M2ENN_Clf.png)

---

## **3️⃣ Final Experiment: SMOTEENN + class_weight=None**
This gave the most stable performance across classes.

### Final Stacking Model Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/M3SMOTE_clf-NE.png)

---

## 🏆 Final Combined Model (Ensemble Fusion)
After evaluating all configurations, I combined **all models** (3 classifiers + 2 boosted models) to create a **majority voting system**:

### **Voting Strategy:**
```
processed_input → each pipeline → each model prediction → inverse transform → majority vote
```

This drastically improved **recall, F1-score, and robustness**.

### Final Combined Performance  
![](https://github.com/KToppo/Emotion-Detection-ML/blob/master/models/final_model.png)

---

## 🖥️ Streamlit Web Application

The application provides two modes:

### ✔ Webcam Emotion Detection  
Runs real-time detection using the browser camera.

### ✔ Image URL Emotion Detection  
User pastes any image URL → model predicts the emotion.

---

## 🚀 How to Run the Project

### **1. Install Dependencies**
```
pip install -r requirements.txt
```

### **2. Run Streamlit App**
```
streamlit run web-app.py
```

### **3. Use the Sidebar to Switch Between**
- Webcam Emotion Detection  
- Image URL Emotion Detection  

---

## 📚 Learnings & Improvements

### ✔ **Improved Data Quality**
- Converting raw images to consistent 48×48 grayscale
- Face detection improved dataset reliability

### ✔ **Tried Multiple Sampling Methods**
- SMOTE improved minority classes
- SMOTEENN removed noisy samples

### ✔ **Model Diversity Helps**
Different models specialize in different emotion classes.

### ✔ **Ensemble Voting**
Combining all models drastically stabilizes predictions.

### ✔ **Modular Architecture**
- Separate pipelines  
- Separate saved models  
- Reusable face detection pipeline  
- Web app integration  

---

## 🎯 Conclusion
This project demonstrates a complete end‑to‑end **Emotion Recognition Pipeline**:
- Dataset creation → Model building → Evaluation → Deployment  
- Multiple experiments to understand data imbalance  
- A powerful ensemble‑based final model  
- Live deployment via Streamlit

This README documents the journey, the models, the improvements, and the reasoning behind the final solution.

---

## 🙌 Author
**Kalyan Toppo**  
Emotion Detection ML System – fully designed, trained & deployed.

---

