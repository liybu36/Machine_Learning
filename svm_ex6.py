'''
Standford Machine Learning Class Ex6
Support Vector Machine
implemented by Hao Qian, Mar 29,2017
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import *
from scipy.io import loadmat
from sklearn.svm import SVC,LinearSVC
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import GridSearchCV,learning_curve,StratifiedShuffleSplit
from docutils.nodes import legend
from sklearn.metrics import confusion_matrix, classification_report

plt.style.use('ggplot')

def plotData(X,y):
    y=y.reshape((y.shape[0],))
    pos = (y==1)
    neg = (y==0)
    plt.scatter(X[pos,0],X[pos,1],marker='+',c='b',s=30,label='Positive')
    plt.scatter(X[neg,0],X[neg,1],marker='o',c='y',s=30,label='Negative')
    plt.xlabel('X Feature 0')
    plt.ylabel('X Feature 1')
    plt.legend(loc='upper left')

def sigmoid(z):
    return 1./(1.+np.exp(-z))

def cost1(z):
    # y=1, theta.T.dot(x)>=1
    #-log(1/(1+e^(-z)))
    h = -1*(sigmoid(z))

def visualizeBoundaryLinear(X, y):
    pass

def gaussianKernel(x1, x2, sigma):
    x1 = x1.reshape((x1.size,))
    x2 = x2.reshape((x2.size,))
    diff = x1 - x2
    innerproduct = diff.dot(diff.T)
    sim = np.exp(-innerproduct/(2*(sigma**2)))
    return sim

def loopoverpair(X,sigma):
    m = X.shape[0]
    K = np.ones((m,m))
    for i in np.arange(m):
        for j in np.arange(i+1,m):
            K[i,j] = gaussianKernel(X[i,:],X[j,:],sigma)
            K[j,i] = K[i,j]
    print(K)
    return K

#plot learning curve
def plot_learning_curve(estimator,title,X,y,train_sizes=np.linspace(0.1,1,5),ylim=None,cv=None,n_jobs=1):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training Sample Size')
    plt.ylabel('Score')
    train_sizes,train_scores,test_scores = learning_curve(estimator,X,y,cv=cv,train_sizes=train_sizes,scoring='accuracy',n_jobs=n_jobs) 
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training Score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='Cross-validation Score')
    plt.legend(loc='best')
    return plt

#main part to execute code
path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex6/ex6/"
filename=path+"ex6data1.mat"
train_data = loadmat(filename)
#print(train_data)
print(train_data.keys())
X = train_data['X']
y = train_data['y']
y = y.reshape((y.shape[0],))
m,n = X.shape
print('X shape ',X.shape)
print('y shape ',y.shape)

#visualizing Data
plt.figure(1)
plotData(X,y)

#implement SVM
#svmTrain(X, y, C)
C=[1,100]
for i in np.arange(len(C)):
    svc_model = LinearSVC(C=C[i])
    svc_model.fit(X,y)
    print(svc_model.coef_)
    plt.figure(i+2)
    plot_decision_regions(X=X, y=y,clf=svc_model, legend=2)
    title= " C: {}".format(C[i])
    plt.title(title)

#define a gaussianKernel
x1_test = np.array([1, 2, 1])
x2_test = np.array([0, 4, -1])
sigma = 2
gaus_test = gaussianKernel(x1_test, x2_test, sigma)
print("gaus test value ", gaus_test)

#load data set 2
filename2=path+"ex6data2.mat"
train_data2 = loadmat(filename2)
#print(train_data2)
print(train_data2.keys())
X2 = train_data2['X']
y2 = train_data2['y']
y2 = y2.reshape((y2.shape[0],))
print('X2 shape ',X2.shape)
print('y2 shape ',y2.shape)
plt.figure(4)
#ax = fig.gca()
#ax.set_xlim(0.,1.)
#ax.set_ylim(0.4,1)
plotData(X2,y2)

#train data set2 with gaussian Kernel
C2 = 1
sigma2 = 0.1
'''
X_pairs = loopoverpair(X2,sigma2)
svc_model2 = LinearSVC(C=C2)
svc_model2.fit(X_pairs,y2)
plt.figure(5)
plot_decision_regions(X=X_pairs, y=y2,clf=svc_model2, legend=2)
plt.title("datasets 2")
'''
svc_model2 = SVC(probability=True)
param_list = {'kernel':['linear','rbf'],'C':[1.0,0.1,0.5,5,10,15,20],'gamma':[0.005,0.01,0.05,0.1,0.5,1,3,5,10,30]}
clf = GridSearchCV(svc_model2,param_list,scoring='accuracy')
clf.fit(X2,y2)
print(clf.best_params_)
print(clf.score(X2,y2))
fig=plt.figure(5)
#ax=fig.gca()
ax = plot_decision_regions(X=X2,y=y2,clf=clf,legend='best')
ax.set_xlim(0.,1.)
ax.set_ylim(0.4,1)
plt.title("datasets 2")
cv = StratifiedShuffleSplit(n_splits=3,test_size=0.25,random_state=7)
plot_learning_curve(clf, 'Data set 2 learning curve', X2, y2,cv=cv)
'''
y_prob = clf.predict_proba(X2)
pos = (y==1)
neg = (y==0)
plt.figure(21)
plt.scatter(X2[pos,0],X2[pos,1],marker='+',c=y_prob[:,1],s=30,label='Positive',cmap='Blues')
plt.scatter(X2[neg,0],X2[neg,1],marker='o',c=y_prob[:,0],s=30,label='Negative',cmap='Reds')
plt.xlabel('X Feature 0')
plt.ylabel('X Feature 1')
plt.legend(loc='upper left')
'''

#load data set 3
filename3=path+"ex6data3.mat"
train_data3 = loadmat(filename3)
#print(train_data3)
print(train_data3.keys())
X3 = train_data3['X']
y3 = train_data3['y']
X3_val = train_data3['Xval']
y3_val = train_data3['yval']
y3 = y3.reshape((y3.shape[0],))
y3_val = y3_val.reshape((y3_val.shape[0],))
print('X3 shape ',X3.shape)
print('y3 shape ',y3.shape)
print('X3_val shape ',X3_val.shape)
print('y3_val shape ',y3_val.shape)
plt.figure(8)
plotData(X3,y3)
plt.title('Train 3 data set X vs y')
plt.figure(9)
plotData(X3_val,y3_val)
plt.title('Val 3 data set X vs y')

svc_model3 = SVC(kernel='rbf')
param_list2 = {'C':[0.01,0.03,0.1,0.3,1,3,10,30],'gamma':[0.01,0.03,0.1,0.3,1,3,10,30]}
clf2 =GridSearchCV(svc_model3,param_list2,scoring='accuracy')
clf2.fit(X3_val,y3_val)
print('clf2 params: ',clf2.best_params_)
print('val train score',clf2.score(X3_val,y3_val))
print('x3 test score ',clf2.score(X3,y3))
fig3 = plt.figure(10)
#ax3 = fig3.gca()
ax3 = plot_decision_regions(X3_val,y3_val,clf2,legend='best')
ax3.set_xlim(-0.6,0.3)
ax3.set_ylim(-0.8,0.6)
plt.title('X3 val decision boundary')
plot_learning_curve(clf2, 'Data set 3 val learning curve', X3, y3,cv=cv)

y3_predict = clf2.predict(X3)
print('confusion matrix in X3 data: \n',confusion_matrix(y3,y3_predict))
#target_names = ['0','1']
print('classification report of X3 val :\n',classification_report(y3,y3_predict))

plt.show()
