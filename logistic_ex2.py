'''
assignment 2 of week 3, Stanford Machine Learning course
https://www.coursera.org/learn/machine-learning/home/week/3
modified by Hao Qian, Mar 22, 2017
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

plt.style.use('ggplot')

#load file into pandas dataframe
def read_data_file(filename):
    columns=['Exam1','Exam2','Admitted']
    df = pd.read_csv(filename,delimiter=",",names=columns)
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    return df

#visualize the data before analysis
def plotData(X,y):
    #X is matrix for scores of exams
    #y is the decision of admission
    pos=y.eq(1)
    neg=y.eq(0)
    pos_line = plt.scatter(X[pos].iloc[:,0],X[pos].iloc[:,1],s=40,marker='+',c='b')
    neg_line = plt.scatter(X[neg].iloc[:,0],X[neg].iloc[:,1],s=30,marker='o',c='y')
    plt.legend((pos_line,neg_line),('Admitted','Not admitted'),scatterpoints=1,
           loc='upper right',fontsize=8)
    

#sigmoid function, 1/(1+e^(-z))    
def sigmoid(Z):
    g = 1./(1.+np.exp(-Z))
    #print(g)
    return g

#calculate cost function    
def costFunction(theta,X,y):    
    m=X.shape[0]
    #theta=np.reshape(theta,(len(theta),1))
    cost=(1./m)*(-y.transpose().dot(np.log(sigmoid(X.dot(theta))))-(1-y).transpose().dot(np.log(1-sigmoid(X.dot(theta)))))
    print("cost shape",cost.shape)
    return cost

def compute_cost(theta,X,y): #computes cost given predicted and actual values
    m = X.shape[0] #number of training examples
    #theta = np.reshape(theta,(len(theta),1))
    #y = reshape(y,(len(y),1))    
    J = (1./m) * (-np.transpose(y).dot(np.log(sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))    
    grad = np.transpose((1./m)*np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]#,grad

#get the gradeint of cost
def gradeint(theta,X,y):
    m=X.shape[0]
    #theta=np.reshape(theta,(len(theta),1))
    grad=(1./m)*((sigmoid(X.dot(theta))-y).transpose().dot(X))
    print("grad shape",grad.shape)
    return grad

def fminunc(theta,X,y,alpha,niter):
    cost_min, grad_min=costFunction(theta,X,y)
    grad=grad_min
    for j in np.arange(niter):
        for i in np.arange(len(theta)):
            theta[i] -= alpha*grad[i]
        cost, grad=costFunction(theta,X,y)
        print(theta,cost,grad)
        if(cost<cost_min):
            cost_min = cost
            grad_min = grad
    return cost_min, grad_min        

#get a map with more features
def mapFeature(X1,X2):
    '''Returns a new feature array with more features, comprising of
     X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..'''
    degree=6
    out=np.ones((X1.size,1))
    X1=np.reshape(X1,(X1.size,1))
    X2=np.reshape(X2,(X2.size,1))
    print(X1.shape,X2.shape)
    for i in np.arange(degree)+1:
        for j in np.arange(i+1):
            r = (X1 ** (i - j)) * (X2 ** j)
            out=np.append(out,r,1)
    print("out shape",out.shape)            
    return out
    
#plot decision tree for 3 or more ndims    
def plotDecisionBoundary(theta,X,y):
    n=X.shape[1]
    if n<=3:
        plot_x = np.array([X[:,1].min()-2,X[:,1].max()+2])
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        boundary=plt.plot(plot_x,plot_y)    
        plt.legend(boundary,'Decision Boundary',scatterpoints=1,loc='upper right',fontsize=8)
    else:
        u=np.linspace(-1,1.5,50)
        v=np.linspace(-1,1.5,50)
        z=np.zeros((len(u),len(v)))
        theta=np.reshape(theta,(theta.size,1))
        print(theta.shape)
        for i in np.arange(len(u)):
            for j in np.arange(len(v)):
                 z[i,j]=mapFeature(np.array(u[i]),np.array(v[j])).dot(theta)
        z=z.transpose()
        print(z.shape)
        plt.contour(u,v,z)
        

#give predicted values of y based on theta from minization of cost
def predict(theta,X):
    m=X.shape[0]
    p=np.zeros((m,1))
    Z=X.dot(theta)
    g=sigmoid(Z)
    print(p.shape,g.shape)
    for i in np.arange(m):
        if g[i]<0.5:
            p[i,0]=0
        else:
            p[i,0]=1
    return p

#get the cost function with regularization
def costFunctionReg(theta,X,y,la):
    m=X.shape[0]
    cost=costFunction(theta,X,y) + (la/(2*m))*(theta.transpose().dot(theta)-theta[0]**2)
    return cost

#get the gradeint of regularizated cost function
def gradeintReg(theta,X,y,la):
    m=X.shape[0]
    grad=gradeint(theta,X,y)
    theta=np.reshape(theta,grad.shape)
    theta[0]=0.
    grad += (la/m)*theta
    return grad
    
#main part of the code execution    
path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex2/ex2/"
df=read_data_file(path+"ex2data1.txt")
len_columns=len(df.columns)-1
X=df.iloc[:,:len_columns]
y=df.iloc[:,len_columns]
plotData(X,y)

#add one (all-ones) column to X
X1=X.values
m=X1.shape[0]
X2=np.ones((m,1))
#X2=np.reshape(X2,(m,1))
X3=np.append(X2,X1,1)
print(X3[:2,:]," X3 shape ",X3.shape)

#change the shape of y
y1=y.values
y1=np.reshape(y1,(m,1))
print(y1[:2,:]," y1 shape ",y1.shape)

#create the initial theta values
n=X3.shape[1]
theta=np.zeros((n,1))
print(theta[:2,:]," theta shape ",theta.shape)

cost=costFunction(theta,X3,y1)
#cost=compute_cost(theta,X3,y)
grad=gradeint(theta,X3,y1)
print(cost,grad)

xopt=fmin_bfgs(costFunction,theta,fprime=gradeint,args=(X3,y),maxiter=400)
print(xopt)

plotDecisionBoundary(xopt,X3,y1)
p=predict(xopt,X3)
print(p)
print('Train Accuracy: %f' % ((y1[(p == y1)].size / float(y1.size)) * 100.0))
plt.show()


#analyze the regularization 
df2=read_data_file(path+"ex2data2.txt")
len_columns2=len(df2.columns)-1
Xreg=df2.iloc[:,:len_columns2]
yreg=df2.iloc[:,len_columns2]
plotData(Xreg,yreg)

#add one (all-ones) column to X
Xreg1=Xreg.values
print(Xreg1.shape)
Xreg3= mapFeature(Xreg1[:, 0], Xreg1[:, 1])
mreg=Xreg3.shape[0]
#Xreg2=np.ones((mreg,1))
#Xreg3=np.append(Xreg2,Xreg4,1)
print(Xreg3[:2,:]," Xreg3 shape ",Xreg3.shape)

#change the shape of y
yreg1=yreg.values
yreg1=np.reshape(yreg1,(mreg,1))
print(yreg1[:2,:]," yreg1 shape ",yreg1.shape)

#create the initial theta values
nreg=Xreg3.shape[1]
thetareg=np.zeros((nreg,1))
print(thetareg[:2,:]," thetareg shape ",thetareg.shape)

la=1
costreg = costFunctionReg(thetareg,Xreg3,yreg1,la)
gradreg = gradeintReg(thetareg,Xreg3,yreg1,la)
print("regularization term cost and grad ",costreg,gradreg)

xopt_reg=fmin_bfgs(costFunctionReg,thetareg,fprime=gradeintReg,args=(Xreg3,yreg,la),maxiter=400)
print(xopt_reg)

plotDecisionBoundary(xopt_reg,Xreg3,yreg1)
p_reg=predict(xopt_reg,Xreg3)
print(p_reg)
print('Train Accuracy: %f' % ((yreg1[(p_reg == yreg1)].size / float(yreg1.size)) * 100.0))
plt.show()

'''
the calculation of matrix is tricy, since the dimension is kind of confusion 
it is important to add extra column to the feature (X) in order to account for the offset theta[0]
the scipy library provides minization functions which acts as fminunc in Octave
'''
