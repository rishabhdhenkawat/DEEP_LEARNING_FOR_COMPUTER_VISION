
import numpy as np
from sklearn.linear_model import SGDClassifier
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


from sklearn.preprocessing import LabelEncoder
# function for loadinf  dataset

def Image_Preprocess(path):
    p=Path(path)
   
    image_data=[]
    
    label=[]
    for lab in p.glob('*'):
        lab_name=str(lab).split('\\')[-1][:-1]
        print(lab_name)
        for item in lab.glob("*.jpg"):
            addre='.\\'+ str(item)
            image=cv2.imread(addre)
            image=cv2.resize(image,(100,100))
            #print(item)
            image_data.append(image)
            label.append(lab_name)
    image_data=np.array(image_data)
    image_data=image_data.reshape((image_data.shape[0],-1))
    return image_data,np.array(label)

#Initialising Aguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",type=str,default="./images",help="# address of dataset")

args=vars(ap.parse_args())



print("[INFO] : Loading Images .................")

x,y=Image_Preprocess(args["dataset"])


le=LabelEncoder()
y=le.fit_transform(y)

(x_train,x_test,y_train,y_test)=train_test_split(x,y,random_state=42,test_size=0.25)


penalty=[None,"l1","l2"]

for p in penalty:
    print(f"[INFO] : Training model with {p} penalty")
    class_=SGDClassifier(penalty=p,random_state=42,loss="log")
    class_.fit(x_train,y_train)
    score=class_.score(x_test,y_test)
    print(f"[INFO] : Model Accuracy : {score*100} %")
