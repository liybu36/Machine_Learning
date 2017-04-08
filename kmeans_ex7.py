'''
Stanford Machine Learning Course, Ex7
KMeans Method
implemented by Hao Qian, Apr 5, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import misc
from  mpl_toolkits.mplot3d import Axes3D

from pca_ex7 import *

plt.style.use('ggplot')

#define functions here
#Finding the closest centroids
def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros((m,1))
    for i in np.arange(m):
        diff = np.zeros(K)
        for j in np.arange(K):
            dist = X[i] - centroids[j]
            diff[j] = dist.dot(dist.T)
            
        idx[i] = diff.argmin() + 1 #get the index of the minimum
    idx = idx.astype(int) 
    idx = idx.reshape(idx.shape[0])   
    return idx    

#compute centroid means
def computeCentroids(X, idx, K):
    m,n = X.shape
    centroids = np.zeros((K,n))
    #idx = idx.reshape(idx.shape[0])
    for i in np.unique(idx):
        centroids[i-1] = X[idx==i].mean(axis=0)
    return centroids

#run the KMeans max_iters times and visualize the progress
def runkMeans(X, initial_centroids,max_iters, plot_progress=False):
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.ones(m)
    for i in np.arange(max_iters):
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        centroids = computeCentroids(X, idx, K)
        
    return centroids, idx    

#visualize the progress of KMeans method
def plotProgresskMeans(X, centroids, previous_centroids, idx, K, i):
    plotDataPoints(X, idx, K)
    plt.scatter(centroids[:,0],centroids[:,1],c='b',marker='x',s=40)
    colors = ['b','r','k']
    for j in np.arange(K):
        x_pos = [previous_centroids[j,0],centroids[j,0]]
        y_pos = [previous_centroids[j,1],centroids[j,1]]
        plt.plot(x_pos,y_pos,color=colors[j])
    print('plotted centroid of iter ',i)

def plotDataPoints(X, idx, K):
    idx = idx.reshape(idx.shape[0])
    plt.scatter(X[:,0],X[:,1],c=idx,alpha=0.5,s=30)
  
#randomly initialize the centroid to avoid the local minimal
def kMeansInitCentroids(X, K):
    m,n = X.shape
    #centroids = np.zeros((K,n))
    centroids = np.random.permutation(X)[:K]
    return centroids
    
#reconstruct pixel
def reconstructPixel(X,centroids,idx):
    m,n = X.shape
    K = centroids.shape[0]
    X_recovered = np.zeros((m,n))
    print(np.unique(idx))
    for i in np.arange(K):
        c_i = (idx==(i+1))
        print(c_i)
        X_recovered[c_i] = centroids[i]
    return X_recovered             
    
#main part to execute code
def main():
    path='/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex7/ex7/'
    file2 = path + 'ex7data2.mat'
    train_data2 = loadmat(file2)
    print(train_data2.keys())
    X = train_data2['X']
    print(X.shape)
    #visualize the data
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1],c='r',marker='o',s=30)
    
    #compute and test the find closest centroids
    K=3
    initial_centroids = np.array([[3, 3],[6, 2],[8, 5]])
    idx = findClosestCentroids(X, initial_centroids)
    print('the centroid of first 3 samples:\n ',idx[:3])
    centroids_test = computeCentroids(X, idx, K)
    print('the centroid means of first 3 samples:\n ',centroids_test[:3])

    #start the KMeans method
    max_iters = 10
    plt.figure(2)
    centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
    
    #load image and compress it with KMeans method
    '''
    A = misc.face()
    image_file2 = path+'bird_small.png'
    misc.imsave(image_file2,A)
    A = misc.imread(image_file2)
'''
    image_data3 = loadmat(path+'bird_small.mat')
    A = image_data3['A']
    plt.figure(3)
    plt.title('Original')
    plt.imshow(A)
    
    A = A/255
    m1,m2 = A.shape[0],A.shape[1]
    A = A.reshape((m1*m2,3))
    print('A shape ',A.shape)
    K3 = 16
    max_iters3 = 10
    initial_centroids3 = kMeansInitCentroids(A, K3)
    centroids3, idx3 = runkMeans(A, initial_centroids3, max_iters3)
    print(centroids3.shape)
    print(idx3.shape)
    idx4 = findClosestCentroids(A, centroids3)
    print(idx4)
    X_recovered = reconstructPixel(A,centroids3,idx4)
    X_recovered = (X_recovered*255).astype(int)
    print(X_recovered)
    X_recovered = X_recovered.reshape((m1,m2,3))
    plt.figure(4)
    plt.title('Compressed')
    plt.imshow(X_recovered)   
    
    #3d visualization
    sel = np.random.permutation(np.arange(m1*m2))[:1000]
    fig = plt.figure(6)
    ax = fig.gca(projection='3d')
    ax.scatter(A[sel,0],A[sel,1],A[sel,2],s=30,c=idx4[sel])
    
    #use PCA to see 2d plots
    fig = plt.figure(7)
    pca_A_norm, pca_mu,pca_sigma = featureNormalize(A)
    pca_U, pca_S = pca(pca_A_norm)
    pca_Z = projectData(pca_A_norm, pca_U, 2)
    plotDataPoints(pca_Z[sel], idx4[sel], K3)

    plt.show()

if __name__=="__main__":
    main()