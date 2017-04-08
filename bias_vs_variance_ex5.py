'''
Machine_Learning_Stanford ex5
Regularized Linear regression and Bias vs Variance
implemented by Hao Qian, Mar 28, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *
from scipy.io import loadmat

plt.style.use('ggplot')

#Regularized linear regression cost function
def linearRegCostFunction(theta,X,y,la):
    m = X.shape[0]
    y = np.reshape(y,(y.shape[0],))
    #theta = np.reshape(theta,(theta.size,1))
    #theta is 2*1 ndims
    h = X.dot(theta) #m*1 ndims
    #h = np.reshape(h,(h.size,1))
    norm_J = 1./(2*m)*(h-y).T.dot(h-y)
    reg_J = la/(2*m)*((theta.T.dot(theta)) - theta[0]*theta[0])
    J = norm_J + reg_J
    return J

#gradient of regularized linear regression
def linearRegGradient(theta,X,y,la):
    m = X.shape[0]
    y = np.reshape(y,(y.shape[0],))
    #y = np.reshape(y,(m,1))
    #theta is 2*1 ndims
    #theta = np.reshape(theta,(theta.shape[0],1))
    h = X.dot(theta) #m*1 ndims
    #h = np.reshape(h,(h.size,1))
    print('h shape ',h.shape)
    print('y shape ',y.shape)
    grad = 1./m*X.T.dot(h-y) + la/m*theta # 2*1 ndims
    grad[0] -= la/m*theta[0]
    return grad

#train linear regression fit
def trainLinearReg(X, y, la):
    initial_theta = np.zeros((X.shape[1],1))
    y = np.reshape(y,(y.shape[0],))
    print(initial_theta.shape)
    xopt = fmin_cg(linearRegCostFunction,initial_theta,fprime=linearRegGradient,args=(X,y,la),maxiter=100)
    print('xopt ',xopt)
    return xopt

#learning Curve
def learningCurve(X, y, Xval, yval,la):
    m = X.shape[0]
    error_train = np.zeros((m,1))
    error_val = np.zeros((m,1))
    for i in np.arange(m-1)+1:
        xopt = trainLinearReg(X[:i,:],y[:i,:],la)
        error_train[i] = linearRegCostFunction(xopt,X[:i,:],y[:i,:],la)
        error_val[i] = linearRegCostFunction(xopt,Xval,yval,la)
    return error_train,error_val

#polynomial features
def polyFeatures(X, p):
    m = X.shape[0]
    X_poly = np.zeros((m,p))
    for i in np.arange(p):
        X_poly[:,[i]] = X**(i+1)

    print("X_poly shape ",X_poly.shape)
    return X_poly

def featureNormalize(X):
    n = X.shape[1]
    mu = np.zeros((1,n))
    sigma = np.zeros((1,n))
    for i in np.arange(n):
        mu[0,i] = X[:,i].mean()
        sigma[0,i] = X[:,i].std()
        X[:,i] = (X[:,i]-mu[0,i])/(sigma[0,i])
    return X, mu, sigma

def ScaleFeature(X,mu,sigma):
    n = X.shape[1]
    for i in np.arange(n):
        X[:,i] = (X[:,i]-mu[0,i])/(sigma[0,i])
    return X

def plotFit(min_x, max_x, mu, sigma, theta, p):
    X = np.arange(min_x-15,max_x+25,0.05)
    X = np.reshape(X,(len(X),1))
    X_poly = polyFeatures(X, p)
    X_poly = ScaleFeature(X_poly,mu,sigma)
    X_poly = np.append(np.ones((X_poly.shape[0],1)),X_poly,1)
    plt.plot(X,X_poly.dot(theta),'b--',linewidth=2)

def PlotLearningCurve(error_train,error_val):
    #poly_error_train,poly_error_val = learningCurve(X_poly2,y,Xval_poly2,yval,0)
    m = error_train.size
    plt.plot(np.arange(m),error_train,'b',linewidth=2,label='Train')
    plt.plot(np.arange(m),error_val,'g',linewidth=2,label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.axis([0,13,0,150])
    plt.legend()


def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    m = len(lambda_vec)
    #lambda_vec = np.reshape(lambda_vec,(len(lambda_vec),1))
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in np.arange(m):
        xopt = trainLinearReg(X, y, lambda_vec[i])
        error_train[i] = linearRegCostFunction(xopt,X,y,lambda_vec[i])
        error_val[i] = linearRegCostFunction(xopt,Xval,yval,lambda_vec[i])

    fig = plt.figure(112)
    plt.plot(lambda_vec,error_train,'b--',linewidth=2,label='Train')
    plt.plot(lambda_vec,error_val,'g--',linewidth=2,label='Cross Validation')
    plt.xlabel("lambda Value")
    plt.ylabel("Error")
    plt.legend()

#main function to execute the MultiClass and Neural Networks
path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex5/ex5/"
filename=path+"ex5data1.mat"
#load pixels into data as dict
train_data = loadmat(filename)
print(train_data.keys())
#training data set
#X is the change of water level
#y is the amount of water flowing out of the dam
X=train_data['X']
y=train_data['y']
print("X train shape ",X.shape," X ",X)
print("y train shape ",y.shape," y ",y)
#cross validation data set
Xval=train_data['Xval']
yval =train_data['yval']
print("X cross shape ",Xval.shape," Xval ",Xval)
print("y cross shape ",yval.shape," yval ",yval)
#test data set
Xtest=train_data['Xtest']
ytest=train_data['ytest']
print("X test shape ",Xtest.shape," Xtest ",Xtest)
print("y test shape ",ytest.shape," ytest ",ytest)

#plot the training data
fig = plt.figure(1)
plt.scatter(X,y,c='r',marker='x',s=40,linewidths=2)
plt.xlabel('Chnage in water level (X)')
plt.ylabel('Water flowing out of the dam (y)')

#calculate regularized linear regression
m = X.shape[0]
X1 = np.append(np.ones((m,1)),X,1)
Xval1 = np.append(np.ones((Xval.shape[0],1)),Xval,1)

theta = np.ones((2,1))
cost = linearRegCostFunction(theta,X1,y,1)
print("cost ",cost)
grad = linearRegGradient(theta,X1,y,1)
print("grad ",grad)

xopt = trainLinearReg(X1, y, 0)
xopt = np.reshape(xopt,(xopt.size,1))
#x_range = np.array([X.min(),X.max()])
y_range = X1.dot(xopt)
plt.plot(X,y_range,'b--',linewidth=2)

#plot the learning Curve
error_train, error_val = learningCurve(X1,y,Xval1,yval,0)
fig = plt.figure(2)
plt.plot(np.arange(m),error_train,'b',linewidth=2,label='Train')
plt.plot(np.arange(m),error_val,'g',linewidth=2,label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,150])
plt.legend()

#polynomial regression
p=8
X_poly = polyFeatures(X, p)
#feature scaling
X_poly1, mu, sigma = featureNormalize(X_poly)
X_poly2 = np.append(np.ones((X_poly1.shape[0],1)),X_poly1,1)

X_poly_opt = trainLinearReg(X_poly2,y,0)
fig = plt.figure(3)
plt.scatter(X,y,c='r',marker='x',s=40,linewidths=2)
plt.xlabel('Chnage in water level (X)')
plt.ylabel('Water flowing out of the dam (y)')
plotFit(X.min(),X.max(),mu,sigma,X_poly_opt,p)

Xval_poly = polyFeatures(Xval,p)
Xval_poly1 = ScaleFeature(Xval_poly,mu,sigma)
Xval_poly2 = np.append(np.ones((Xval_poly1.shape[0],1)),Xval_poly1,1)
print("Xval_poly2 shape ",Xval_poly2.shape)
Xtest_poly = polyFeatures(Xtest,p)
Xtest_poly1 = ScaleFeature(Xtest_poly,mu,sigma)
Xtest_poly2 = np.append(np.ones((Xtest_poly1.shape[0],1)),Xtest_poly1,1)
print("Xtest_poly2 shape ",Xtest_poly2.shape)

#lambda = 0
poly_error_train,poly_error_val = learningCurve(X_poly2,y,Xval_poly2,yval,0)
fig = plt.figure(4)
PlotLearningCurve(poly_error_train,poly_error_val)
'''
plt.plot(np.arange(m),poly_error_train,'b',linewidth=2,label='Train')
plt.plot(np.arange(m),poly_error_val,'g',linewidth=2,label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,150])
plt.legend()
'''
#see how the fitting chanes as lambda term
laval = [0,1,10,100]
for i in np.arange(len(laval)):
    title = "lambda is %d" % laval[i]
    X_poly_opt_temp = trainLinearReg(X_poly2,y,laval[i])
    fig = plt.figure(5+2*i)
    plt.title(title)
    plt.scatter(X,y,c='r',marker='x',s=40,linewidths=2)
    plt.xlabel('Chnage in water level (X)')
    plt.ylabel('Water flowing out of the dam (y)')
    plotFit(X.min(),X.max(),mu,sigma,X_poly_opt_temp,p)

    poly_error_train_temp,poly_error_val_temp = learningCurve(X_poly2,y,Xval_poly2,yval,1)
    fig = plt.figure(5+2*i+1)
    plt.title(title)
    PlotLearningCurve(poly_error_train_temp,poly_error_val_temp)

#cross Validation curve
validationCurve(X_poly2,y,Xval_poly2,yval)

plt.show()
