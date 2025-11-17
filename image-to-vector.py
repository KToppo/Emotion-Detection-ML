import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

path = 'Assets/Data'
emotions = os.listdir(path)
# x = pd.DataFrame(columns=[f'{i}_px' for i in range(2304)])
# y = pd.DataFrame(columns=['emotion'])
N = 0
for e in emotions:
    N += len(os.listdir(f'Assets/Data/{e}'))

features = np.zeros((N,48*48), dtype=np.uint8)
labels = []
i=0
for emo in emotions:
    imgs = os.listdir(f'{path}/{emo}')
    print(f'Working on {emo}')
    for img in tqdm(imgs):
        img_ = Image.open(f"{path}/{emo}/{img}")
        img_ = img_.resize((48, 48)).convert('L')
        img_array = np.array(img_)

        features[i,:] = img_array.ravel()
        labels.append(emo)
        i+=1
        # sr = pd.Series(img_array.ravel(), index=x.columns)

        # x = pd.concat([x, sr.to_frame().T], ignore_index=True)
        # y = pd.concat([y,pd.DataFrame({'emotion':emo}, index=[0])], ignore_index=True)
        

x = pd.DataFrame(features, columns=[f'{i}_px' for i in range(2304)])
x['emotion'] = labels
x.drop_duplicates(inplace=True)


# x['emotion'] = y['emotion']
x.to_csv('data.csv',index=False)




# imgs = os.listdir(f'{path}{emotions[0]}')
# # print(imgs)
# img = Image.open(f"{path}{emotions[0]}/{imgs[0]}") 
# img_array = np.array(img)
# sr = pd.Series(img_array.ravel(), index=x.columns)


# x = pd.concat([x, sr.to_frame().T], ignore_index=True)
# # print(img_array.ravel().shape)
# img = Image.open(f"{path}{emotions[0]}/{imgs[1]}") 
# img_array = np.array(img)
# sr = pd.Series(img_array.ravel(), index=x.columns)


# x = pd.concat([x, sr.to_frame().T], ignore_index=True)
# print(x)