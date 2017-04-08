'''
Stanford Machine Learning Course, ex8
Anomaly detection system
implemented by Hao Qian, Apr 6, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
import pandas as pd

plt.style.use('ggplot')

#function definition
#estimate parameters for Gaussian distribution
def estimateGaussian(X):
    m,n = X.shape
    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)
    print('mu shape ',mu.shape,' sigma2 shape ',sigma2.shape)
    print('mu is:\n',mu,'\n sigma^2 is \n',sigma2)
    return mu, sigma2

#Multivariate Gaussian probability
def multivariateGaussian(X, mu, Sigma2):
    #here X must be 1d array, like (n,) shape
    X = X.reshape(mu.shape)
    n = mu.size
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)
    diff = (X - mu).reshape((n,1)) 
    #print('diff \n',diff.dot(diff.T))
    #print(Sigma2.shape)
    det = np.linalg.det(Sigma2)
    inv_Sigma2 = np.linalg.inv(Sigma2)
    #prob = 1./(np.power(2*np.pi,n/2)*np.sqrt(det))*np.exp(-0.5*diff.T*inv_Sigma2*diff)  
    prob = 1./(np.power(2*np.pi,n/2)*np.sqrt(det))*np.exp(-0.5*diff.T.dot(inv_Sigma2.dot(diff)))    
    #print('prob is :',prob)
    return prob

def multidimsGaussian(X,mu,Sigma2):
    #X can be (m,n) shape 
    m,n = X.shape
    prob = np.zeros(m)
    for i in np.arange(m):
        #print(X[i], X[i].shape)
        prob[i] = multivariateGaussian(X[i], mu, Sigma2)
    
    print('multidims Gaussian dims ',prob.shape)
    return prob

#visualize the 2d contour of the gaussian fit
def visualizeFit(X, mu, sigma2):
    a = np.arange(0,35,0.5)
    X1, X2 = np.meshgrid(a,a)
    X11 = X1.flatten('F')
    X22 = X2.flatten('F')
    Z = multidimsGaussian(np.matrix([X11,X22]).T,mu,sigma2)
    Z = Z.reshape(X1.shape)
    plt.scatter(X[:,0],X[:,1],c='b',marker='x',s=30)
    plt.contour(X1,X2,Z,levels=10.**(np.arange(-20,0,3)))    
 
def predictAnomalous_detection(pval,epi):
    if pval < epi:
        #y=1 is anomalous
        return 1
    else:
        #y=0 is normal sample
        return 0
 
#select the threshold of eplison to detect the anomaly 
def selectThreshold(yval, pval):
    yval = yval.reshape((yval.size,))
    pval = pval.reshape((pval.size,))
    m = yval.size
    bestEpsilon = 0
    bestF1 = 0    
    N = 1000
    step = np.linspace(pval.min(),pval.max(),N)
    for epi in step:
        predict_y = np.zeros(m)
        for i in np.arange(m):
            predict_y[i] = predictAnomalous_detection(pval[i], epi)
        tp = ((yval==1) & (predict_y==1)).sum()    
        fp = ((yval==0) & (predict_y==1)).sum()   
        fn = ((yval==1) & (predict_y==0)).sum()   
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)
        if F1>bestF1:
            bestF1 = F1
            bestEpsilon = epi
            
    print('best epi: ',bestEpsilon,' best F1: ',bestF1)
    return bestEpsilon,bestF1        
    
#convert array to pandas DataFrame ans draw PairGrid
def DrawPairGrid(X):
    n = X.shape[1]
    print(X.shape)
    columns = ['X%s' % i for i in np.arange(n)]
    df = pd.DataFrame(X,columns=columns)
    #plt.figure()
    g = sns.PairGrid(df)
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter) 
    print(df.corr())
    plt.figure()
    plt.imshow(df.corr(),cmap=plt.cm.RdYlGn)
    plt.colorbar()
    
def covarianceMatrix(X,mu,Sigma):
    m,n = X.shape
    mux, muy = np.meshgrid(mu,np.arange(m))
    #print(mu)
    #print(mux,' mux shape',mux.shape)
    diff = X-mux
    covariancematrix = (1./m)*(X-mux).T.dot(X-mux)
    print('covariance matrix ',covariancematrix)
    Sigma = np.sqrt(Sigma)
    Sigma = Sigma.reshape((1,n))
    sig_x_sig_y = Sigma.T.dot(Sigma)
    corr_matrix = np.divide(covariancematrix,sig_x_sig_y)
    print('corr matrix',corr_matrix)
    

#main code to execute 
def main():
    path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex8/ex8/"
    file1 = path+'ex8data1.mat'
    train_data1 = loadmat(file1)
    print('train data 1 keys:\n',train_data1.keys())
    X = train_data1['X']
    Xval = train_data1['Xval']
    yval = train_data1['yval']
    print('train data 1 X shape: ',X.shape,' X sample \n',X[:5])
    print('corss validation Xval shape {}, yval shape {}'.format(Xval.shape,yval.shape))
    
    #visualize the data 
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1],c='b',marker='x',s=30,label='Train Data 1')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.legend()

    #calculate mu, sigma2 using gaussian distribution
    mu, Sigma2 = estimateGaussian(X)
    #p = multivariateGaussian(X, mu, sigma2)
    #visualize the gaussian fit
    plt.figure(2)
    visualizeFit(X, mu, Sigma2)
    
    #try Xval and yval
    #print(Xval)
    pval = multidimsGaussian(Xval,mu,Sigma2)
    epsilon, F1 = selectThreshold(yval,pval)
    outlier = (pval < epsilon)
    plt.scatter(X[outlier,0],X[outlier,1],c='r',marker='o',s=30)
    covarianceMatrix(X,mu,Sigma2)
    DrawPairGrid(X)

    #load a much larger dataset 
    file2 = path+'ex8data2.mat'
    train_data2 = loadmat(file2)
    print(train_data2.keys())
    X2 = train_data2['X']
    Xval2 = train_data2['Xval']
    yval2 = train_data2['yval']
    print('X2 shape ',X2.shape,' Xval2 shape: ',Xval2.shape)
    mu2, Sigma22 = estimateGaussian(X2)
    p = multidimsGaussian(X2, mu2, Sigma22)
    pval2 = multidimsGaussian(Xval2, mu2, Sigma22)
    epsilon2, F12 = selectThreshold(yval2,pval2)
    outlier2 = (pval2 < epsilon2).sum()
    print('cross validation anomous sets: ',outlier2)
    print('train set anomous sets: ', (p<epsilon2).sum())
    covarianceMatrix(X2,mu2,Sigma22)
    DrawPairGrid(X2)

    '''
    plt.figure(5)
    plt.scatter(X2[:,0],X2[:,1],c='b',marker='x',s=30)
    plt.title('larger train set')
    
    plt.figure(6)
    plt.scatter(Xval2[:,0],Xval2[:,1],c='b',marker='x',s=30)
    outlier2 = (pval2 < epsilon2)
    plt.scatter(Xval2[outlier2,0],Xval2[outlier2,1],c='r',marker='o',s=30)
    '''

    plt.show()

if __name__=='__main__':
    main()