from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    #compute the sigmoid activation value for a given input
    return 1/(1+np.exp(-x))
inp=np.linspace(-10,10,100)
plt.plot(inp,sigmoid_activation(inp))
plt.show()
def predict(x,w):
    preds=sigmoid_activation(x.dot(w))
    preds[preds<=0.5]=0
    preds[preds>0]=1
    
    return preds

#construct he argument parse and parse the argument
ap=argparse.ArgumentParser()
ap.add_argument("-e","--epochs",type=float,default=100,help="# of epochs")
ap.add_argument("-a","--alpha",type=float,default=0.01,help="learning rate")
args=vars(ap.parse_args())


(x,y)=make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)
y=y.reshape((y.shape[0],1))
plt.scatter(x[:,0],x[:,1],color="blue")
plt.show()
x=np.c_[x,np.ones((x.shape[0]))]
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.5,random_state=42)

#initialise our weight matrix and list of losses
print("[INFO] training...")
W=np.random.randn(x.shape[1],1)
losses=[]
for epoch in np.arange(0,args["epochs"]):
    preds=sigmoid_activation(trainx.dot(W))
    error=preds-trainy
    loss=np.sum(error**2)
    losses.append(loss)
    gradient =trainx.T.dot(error)
    W+=-args["alpha"]*gradient 
    #check if see if an update should be displayed
    if epoch==0 or (epoch+1)%5==0:
        print("[INFO] epoch={},loss={:.7f}".format(int(epoch+1),loss))
preds=predict(testx,W)
print(classification_report(testy,preds))
cm=confusion_matrix(testy,preds)
    
#Plot the classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testx[:,0],testx[:,1],marker="o",s=30)

#Construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,args["epochs"]),losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()