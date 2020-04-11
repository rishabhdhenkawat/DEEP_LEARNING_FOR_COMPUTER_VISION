#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report ,confusion_matrix
from sklearn.model_selection import train_test_split

def sigmoid(z): #sigmoid Function
    return 1/(1+np.exp(-z))
def hypothesis(x,w): # Hypothesis Function for Predictions
    return sigmoid(np.dot(x,w))
def loss(x,w,y):    # Loss funcions
    y_=hypothesis(x,w)
    return np.mean(-y*np.log(y_)-(1-y)*np.log(1-y_))
    
def update(x,y,w,learning_rate):  #Update rule for weights
    y_=hypothesis(x,w)
    dw=np.dot(x.T,(y_-y))
    
    w=w-learning_rate*dw/(float(x.shape[0]))
    return w
def train(x,y,learning_rate=0.1,maxItr=100):  # Traing our model
    losses=[]
    ones=np.ones((x.shape[0],1))
    x=np.hstack((ones,x))
    w=np.zeros((x.shape[1]))
    for mi in range(maxItr):
        l=loss(x,w,y)
        losses.append(l)
        w=update(x,y,w,learning_rate)
        if (mi+1)%5 ==0 :
            print(f"[INFO]: epoch:{mi+1}  loss :{l}")
    return losses,w


def predictions(x,w):    #making Predictions
    if x.shape[1] != w.shape[0]:
        ones = np.ones((x.shape[0],1))
        x = np.hstack((ones,x))
    pred=hypothesis(x,w)
    pred[pred>=0.5]=1
    pred[pred<0.5]=0
    return pred


#Creating Data
X,y=make_blobs(n_samples=500,
    n_features=2,
    centers=2,
    cluster_std=1.5,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=42,
)

print(f"Shape of feature matrix : {X.shape} and shape of label vector{y.shape}")





plt.style.use("seaborn")

#Visualising our Data

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("feature 1")
plt.ylabel("feature2")
plt.show()



x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.25)

#Initialising Aguments
ap=argparse.ArgumentParser()
ap.add_argument("-e","--maxItr",type=int,default=100,help="# no of Iteratins")
ap.add_argument("-a","--learning_rate",type=float,default=0.01,help="# Learning_Rate")


args=vars(ap.parse_args())


print("[INFO] : Evaluating........")
print("[INFO] : Training........")
l,w=train(x_train,y_train,maxItr=args["maxItr"],learning_rate=args["learning_rate"])


plt.plot(l,c="r",label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


Y=predictions(x_test,w)

print("[INFO] : Calculating accuracy,precision and recall.............")

print(classification_report(y_test,Y))

print("[INFO] : Confusion Matrix ")
print(confusion_matrix(y_test,Y))

#Visualising the Predicted Line

plt.scatter(X[:,0],X[:,1],c=y,)
plt.xlabel("feature 1")
plt.ylabel("feature2")
sample=np.linspace(5,-5,50)
x2=-(w[0]*1+w[1]*sample)/w[2]
plt.plot(sample,x2,c="r",label="Predicted Line")
plt.legend()
plt.show()




