'''
MultiClass class and Neural Network
Ex3 of Machine Learning Class Stanford
implemented by Hao Qian, Mar 24, 2017
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import *

plt.style.use('ggplot')

#display the image data sample
def displayData(X, example_width):
    m,n = X.shape
    if(example_width<1):
        example_width = np.round(np.sqrt(n))
    example_height = n/example_width
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m/display_rows)
    pad = 1
    display_array = np.zeros((pad+display_rows*(example_height+pad),pad+display_cols*(example_width+pad)))

    curr_ex = 0
    for j in np.arange(display_rows):
        for i in np.arange(display_cols):
            if curr_ex>m:
                break
            max_val=np.abs(X[curr_ex,:]).max()
            pad_x=pad+j*(example_height+pad)
            pad_y=pad+i*(example_width+pad)
            #display_array[pad+j*(example_height+pad)+(1:example_height),pad+i*(example_width+pad)+(1:example_width)]=np.reshape(X[curr_ex,:],(example_height,example_width))/max_val
            display_array[pad_x:(pad_x+example_height),pad_y:(pad_y+example_width)]=np.reshape(X[curr_ex,:],(example_height,example_width))/max_val
            curr_ex += 1
        if curr_ex>m:
            break

    plt.imshow(display_array,extent=[0,10,0,10],cmap=plt.cm.Greys_r)

#calculate logistic function
def sigmoid(z):
    g = 1./(1.+np.exp(-z))
    return g

#calculate vectorized cost function
def computeCost(theta,X,y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    J=(1./m)*(-y.transpose().dot(np.log(h))-(1-y).transpose().dot(np.log(1-h)))
    return J

#calculate gradient function of J
def gradient(theta,X,y):
    m=X.shape[0]
    h = sigmoid(X.dot(theta))
    grad=(1./m)*((h-y).transpose().dot(X))
    grad=grad.transpose()
    return grad

#calculate regularized logistic regression cost function
def computesCostReg(theta,X,y,la):
    m=X.shape[0]
    J = computeCost(theta,X,y) + la/(2*m)*(theta.transpose().dot(theta)-theta[0]**2)
    return J

#calculate regularized gradient
def gradientReg(theta,X,y,la):
    m=X.shape[0]
    grad = gradient(theta,X,y)+(la/m)*theta
    grad[0] -= (la/m)*theta[0]
    return grad

#oneVsAll classifiers for multi classifiers
def oneVsAll(X, y, num_labels, la):
    m,n = X.shape
    y_labels = np.unique(y)
    #number of classes K
    K = y_labels.size
    all_theta = np.zeros((K,n))

    for i in np.arange(K):
        y_tag = np.zeros(m) #do not use (m,1) as the dim of y_tag
        theta = np.zeros((n,1))
        for j in np.arange(m):
            if y[j] == y_labels[i]:
                y_tag[j] = 1

        xopt=fmin_cg(computeCost,theta,fprime=gradient,args=(X,y_tag),maxiter=50) # dim of y is (m,)
        #xopt=minimize(computeCost,theta,method='TNC',jac=gradient,args=(X,y_tag))
        print(xopt)
        #xopt=np.reshape(xopt.x,(1,n))
        xopt=np.reshape(xopt,(1,n))
        all_theta[i]=xopt
    print("all_theta shape ",all_theta.shape)
    return all_theta

#predict function
def predict(theta,X):
    m=X.shape[0]
    g=sigmoid(X.dot(theta))
    p=np.zeros(m)
    for i in np.arange(m):
        if g[i] < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    return p

#calculate the Accuracy of the predictions
def predictOneVsAll(all_theta, X):
    m,n = X.shape
    K = all_theta.shape[0]
    #g = sigmoid(X.dot(all_theta.T)).transpose()
    g = X.dot(all_theta.T).transpose()
    #p = np.zeros((k,m))
    #for i in np.arange(K):
    #    p[i] = predict(all_theta[i],X)
    h = np.zeros(m)
    for i in np.arange(m):
        for j in np.arange(K):
            if g[j,i] == g[:,i].max():
                h[i] = j+1
    h=np.reshape(h,(m,1))
    return h

#map y to k*m dimension data
def mapy(y):
    y_labels = np.unique(y)
    K = y_labels.size
    m = y.shape[0]
    y_map = np.zeros((K,m))
    for i in np.arange(K):
        y_tag = np.zeros(m) #do not use (m,1) as the dim of y_tag
        for j in np.arange(m):
            if y[j] == y_labels[i]:
                y_tag[j] = 1
        y_map[i] = y_tag
    return y_map

#prediction function for Neural Networks
def NeuralPredict(N_a3):
    K,m = N_a3.shape
    p = np.zeros((K,m))
    for i in np.arange(K):
        for j in np.arange(m):
            if N_a3[i,j] < 0.5:
                p[i,j] = 0
            else:
                p[i,j] = 1
    return p

def NeuralFeedForward(Theta1,Theta2,X):
    m = X.shape[0]
    K = Theta2.shape[0]
    a2 = sigmoid(Theta1.dot(X.transpose())) #25*m dimensions
    a2_0 = np.ones((1,m))
    a2 = np.append(a2_0,a2,0)
    a3 = sigmoid(Theta2.dot(a2)) #10*m dimensions
    h = np.zeros(m)
    for i in np.arange(m):
        for j in np.arange(K):
            if a3[j,i] == a3[:,i].max():
                h[i] = j+1
    h=np.reshape(h,(m,1))
    return h

#main function to execute the MultiClass and Neural Networks
path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex3/ex3/"
filename=path+"ex3data1.mat"
#load pixels into data as dict
train_data = loadmat(filename)
print(train_data.keys())
X=train_data['X']
y=train_data['y']
print("X shape ",X.shape," y shape ",y.shape)
m,n = X.shape

#random select a list from X and display the handwritten
#rand_indices = np.arange(m)
#rand_indices = np.random.permutation(rand_indices)
rand_indices = np.random.permutation(m)
displayData(X[rand_indices[:100],:],0)

#Vectorize logistic regression
X1=np.ones((m,1))
X2=np.append(X1,X,1)
print("X2 shape ",X2.shape)
y_map = mapy(y)
print('y ',y)
print('y_map ', y_map)
print('y_map shape',y_map.shape)

#one for all classifiers
num_labels = 10
la=0.1

all_theta = oneVsAll(X2, y, num_labels, la)
print("all_theta shape ",all_theta.shape)
print(all_theta)

predict = predictOneVsAll(all_theta, X2)
print('predict shape ',predict.shape,' y_map shape ',y_map.shape)
print('MultiClass Train Accuracy: %f' % ((y[(predict == y)].size / float(y.size)) * 100.0))

#Neural Networks
weightfilename = path+'ex3weights.mat'
weight=loadmat(weightfilename)
print(weight.keys())
Theta1 = weight['Theta1']
Theta2 = weight['Theta2']
print("Theta1 shape ",Theta1.shape)
print("Theta2 shape ",Theta2.shape)

N_a3 = NeuralFeedForward(Theta1,Theta2,X2)
#N_predict = NeuralPredict(N_a3)
print('N_a3 ',N_a3)
#print('N_predict[0] ',N_predict[0])
#print('Neural Network Train Accuracy: %f' % ((y_map[(N_predict == y_map)].size / float(y_map.size)) * 100.0))
print('Neural Network Train Accuracy: %f' % ((y[(N_a3 == y)].size / float(y.size)) * 100.0))

plt.show()
