__version__ = "2.0.0"

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def adjusted_detect_faces(img):
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    crops = []
    for (x,y,w,h) in faces:
        crops.append(img[y:y+h, x:x+w])
    return crops


path = "Assets/Data"
emotions = os.listdir(path)

BATCH_SIZE = 500
batch_features = []
batch_labels = []

csv_file = "data.csv"
header_written = False

for emo in emotions:
    imgs = os.listdir(f"{path}/{emo}")
    print(f"Working on {emo}...")

    for img_name in tqdm(imgs):
        
        # Load image
        img_path = f"{path}/{emo}/{img_name}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Detect faces
        h, w = img.shape

        if (h, w) == (48, 48):
            crops = [img]
        else:
            crops = adjusted_detect_faces(img)

        del img

        
        if not crops:
            continue

        # Process crops
        for c in crops:
            crop48 = cv2.resize(c, (48,48))
            batch_features.append(crop48.ravel())
            batch_labels.append(emo)
        del crops  # free
        del c
        gc.collect()

        # Save batch
        if len(batch_features) >= BATCH_SIZE:
            df = pd.DataFrame(batch_features,columns=[f'{i}_px' for i in range(2304)])
            df["emotion"] = batch_labels

            df.to_csv(csv_file, mode='a', header=not header_written, index=False)
            header_written = True

            batch_features.clear()
            batch_labels.clear()
            gc.collect()

# Save last batch
if batch_features:
    df = pd.DataFrame(batch_features,columns=[f'{i}_px' for i in range(2304)])
    df["emotion"] = batch_labels
    df.to_csv(csv_file, mode='a', header=not header_written, index=False)

print("Saved all chunks!")