'''
Machine_Learning_Stanford ex1
Linear regression
implemented by Hao Qian, Mar 20, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_bfgs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,ticker,colors
from numpy.linalg import inv

plt.style.use('ggplot')

#get the ndim*ndim identity matrix
def warmUpExercise(ndim):
    return np.eye(ndim)

#load data files
def loadData(filename):
    columns=['population','profit']
    df=pd.read_csv(filename,names=columns,delimiter=',')
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    return df

#visualize data
def plotData(X,y):
    plt.scatter(X,y,marker='x',c='r',s=40)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")

#get the hypothesis h(x)
def hypo(X,theta):
    return X.dot(theta)

#get the cost function J
def computeCost(X,y,theta):
    m=X.shape[0]
    h=hypo(X,theta)
    delta = h-y
    print(delta.shape)
    J=(1./(2*m))*(delta.transpose().dot(delta))
    return J

#get the gradient function
def gradient(X,y,theta):
    m=X.shape[0]
    h=hypo(X,theta)
    delta = h-y
    grad=(1./m)*(X.transpose().dot(delta))
    return grad

#loop over to minimize cost function
def gradientDescent(X,y,theta,alpha,num_iters):
    J_history=np.zeros((num_iters,1))
    cost_min = computeCost(X,y,theta)
    theta_min = theta
    for i in np.arange(num_iters):
        theta -= alpha*gradient(X,y,theta)
        cost = computeCost(X,y,theta)
        J_history[i]=cost
        print("iter ",i," cost ",cost)
        if(cost_min>cost):
            cost_min = cost
            theta_min = theta
    print("minized cost ",cost_min," theta_min ",theta_min)
    return theta_min, J_history

#plot 2-D cost function J vs theta
def plot_J_vs_theta(X,y,theta_min):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    M = theta0_vals.size
    N = theta1_vals.size
    J_vals = np.zeros((M,N))
    for i in np.arange(M):
        for j in np.arange(N):
            t=np.array([theta0_vals[i],theta1_vals[j]])
            t=np.reshape(t,(X.shape[1],1))
            J_vals[i,j]=computeCost(X,y,t)
    fig = plt.figure(1)
    theta0, theta1 = np.meshgrid(theta0_vals,theta1_vals)
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta0,theta1,J_vals,cmap=cm.coolwarm)
    fig2=plt.figure(2)
    plt.contour(theta0_vals,theta1_vals,J_vals)
    plt.plot(theta_min[0],theta_min[1],c="r",marker='x',markersize=4)
    plt.xlabel("\theta_0")
    plt.ylabel("\theta_1")
    #plt.contour(theta0_vals,theta1_vals,J_vals,locator=ticker.LogLocator())
    # new_contur=fig.add_subplot(212)
    # new_contur.contour(theta0,theta1,J_vals)


#the main part to execuste code for linear regression
path='/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex1/ex1/'
linear_filename=path+'ex1data1.txt'
df=loadData(linear_filename)
len_columns=len(df.columns)-1
X=df.iloc[:,:len_columns]
y=df.iloc[:,len_columns]
X=X.values
y1=y.values
m=X.shape[0]
y1=np.reshape(y1,(m,1))
print(m," X shape ",X.shape)
print("y1 shape ",y1.shape)
plotData(X,y1)

#add one all-ones column to X
X1=np.ones((m,1))
X2=np.append(X1,X,1)
print("X2 shape ",X2.shape)
n=X2.shape[1]

theta=np.zeros((n,1))
print("theta shape", theta.shape)
iterations = 1500;
alpha = 0.01;
cost=computeCost(X2,y1,theta)
print(cost," cost theta 0")
alpha_test=np.array([-1,2])
alpha_test=np.reshape(alpha_test,(n,1))
cost1=computeCost(X2,y1,alpha_test)
print(cost1," test theta cost ")

theta_min,J_history=gradientDescent(X2,y1,theta,alpha,iterations)
predict1 = np.array([1,3.5]).dot(theta_min)
predict2 = np.array([1,7]).dot(theta_min)
print("predict1 ",predict1," predict2 ",predict2)
plt.plot(X2[:,1],X2.dot(theta_min),c='b')
plt.show()

plot_J_vs_theta(X2,y1,theta_min)
#xopt=fmin_bfgs(computeCost,theta,fprime=gradient,args=(X2,y1),maxiter=400)
#print(xopt)
#print('Train Accuracy: %f' % ((y1[(p == y1)].size / float(y1.size)) * 100.0))

#functions for multiple variables
def featureNormalize(X):
    m,n=X.shape
    X_mean = np.zeros((1,n))
    X_std = np.zeros((1,n))
    print(X[:4,:])
    for i in np.arange(n):
        mean = X[:,i].mean()
        std = X[:,i].std()
        X_mean[0,i]=mean
        X_std[0,i]=std
        X[:,i] = 1.*(X[:,i]-mean)/(std)
    print(X[:4,:])
    return X, X_mean, X_std

#Linear regression with multiple variables
multiple_filename=path+'ex1data2.txt'
columns=['size','bedrooms','price']
df2=pd.read_csv(multiple_filename,names=columns)
df2=df2.astype(float)
print(df2.head())
print(df2.describe())
print(df2.dtypes)

len_multi=len(df2.columns)-1
X_multi=df2.iloc[:,:len_multi]
y_multi=df2.iloc[:,len_multi]
X_multi=X_multi.values
y_multi1=y_multi.values
m_multi=X_multi.shape[0]
#change the dtype of array: a=a.astype(float)
y_multi1=np.reshape(y_multi1,(m_multi,1))
print(m_multi," X_multi shape ",X_multi.shape)
print("y_multi1 shape ",y_multi1.shape)

X_multi1,X_multi_mean,X_multi_std = featureNormalize(X_multi)
#add one all-ones column to X
X_multi0=np.ones((m_multi,1))
X_multi2=np.append(X_multi0,X_multi1,1)
print("X_multi2 shape ",X_multi2.shape)
n_multi=X_multi2.shape[1]

theta_multi = np.zeros((n_multi,1))
print("theta_multi shape", theta_multi.shape)
cost_multi=computeCost(X_multi2,y_multi1,theta_multi)
print(cost_multi," cost theta 0")

#try minimize cost by different learning rate
fig3=plt.figure(3)
num_iters = 4000;
'''
alpha = [0.03,0.01]
colors = ['g','black']
plots = []
for i in np.arange(len(alpha)):
    theta_multi_min,J_multi_hist=gradientDescent(X_multi2,y_multi1,theta_multi,alpha[i],num_iters)
    temp_plot = plt.plot(np.arange(num_iters),J_multi_hist,c=colors[i],label=alpha[i])
    plots.append(temp_plot)
plt.legend(plots,,scatterpoints=1,loc='upper right',fontsize=8)
'''
alpha=0.01
theta_multi_min,J_multi_hist=gradientDescent(X_multi2,y_multi1,theta_multi,alpha,num_iters)
print("theta multi min ",theta_multi_min)
plt.plot(np.arange(num_iters),J_multi_hist,c='g')
plt.xlabel('Number of iterations')
plt.ylabel('Cost Function J')

#predict the value
def normalize_Test_X(X_test,X_mean,X_std):
    n=X_mean.shape[1]
    X_test = np.reshape(X_test,(1,n))
    X_Norm_test=np.zeros((1,n))
    for i in np.arange(n):
        X_Norm_test[0,i] = (X_test[0,i]-X_mean[0,i])/X_std[0,i]
    return X_Norm_test


predict_X = np.array([1650.,3.])
predict_X_test = normalize_Test_X(predict_X,X_multi_mean,X_multi_std)
predict_X_test = np.append(1,predict_X_test)
print(predict_X_test.shape, theta_multi_min.shape)
predict1 = predict_X_test.dot(theta_multi_min)
print("predict 1 ",predict1)

#Normal Equations
def normalEqn(X, y):
    m,n = X.shape
    theta = np.zeros((n,1))
    det = X.transpose().dot(X)
    X_inv = inv(det)
    normal_theta = X_inv.dot(X.transpose().dot(y))
    return normal_theta

normal_theta = normalEqn(X_multi2,y_multi1)
print(normal_theta)
predict2 = predict_X_test.dot(normal_theta)
print("predict 2 ",predict2)

plt.show()
