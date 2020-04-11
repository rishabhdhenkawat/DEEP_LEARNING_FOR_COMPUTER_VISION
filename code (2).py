#!/usr/bin/env python
# coding: utf-8

# In[3]:



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    #compute the sigmoid activation value for a given input
    return 1.0/(1+ np.exp(-x))

def predict(x,W):
    #take the dot product between our feature and weight matrix
    preds= sigmoid_activation(x.dot(W))
    preds[preds<=0.5]=0
    preds[preds>0]=1
    
    #return the predictions
    return preds
#while True:
#    Wgradient=evaluate_gradient(loss,data,W)
#    W += -alpha * Wgradient
ap=argparse.ArgumentParser()
ap.add_argument("-e","--epochs",type=float,default=100,help="# of epochs")
ap.add_argument("-a","--alpha",type=float,default=0.01,help="learning rate")
args=vars(ap.parse_args())
#generate a 2-classclassificataion priblem with 1000 data points
#where each data point is a 2d feature vector
(x,y)=make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)
y=y.reshape((y.shape[0],1))
x=np.c_[x,np.ones((x.shape[0]))]
(trainX,testX,trainY,testY)=train_test_split(x,y,test_size=0.5,random_state=42)
#initialize our weight matrix and list of losses
print("[INFO] training....")
W=np.random.randn(x.shape[1],1)
losses=[]
#loop over the desired number of epochs
for epoch in np.arange(0,args["epochs"]):
    preds=sigmoid_activation(trainX.dot(W))
    error=preds - trainY
    loss=np.sum(error**2)
    losses.append(loss)
    gradient=trainX.T.dot(error)
    W+=-args["alpha"]*gradient
    if epoch ==0 or (epoch+1)%5==0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch+1),loss))
#evaluating the model
print("[INFO] evaluating...")
preds=predict(testX,W)
print(classification_report(testY,preds))
#plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,0],testX[:,1],marker="o",s=30)
#construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,args["epochs"]),losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()


# In[ ]:




