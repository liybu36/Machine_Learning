'''                                                                                                                                        
Stanford Machine Learning Course, Ex7                                                                                                      
PCA Method
implemented by Hao Qian, Apr 5, 2017                                                                                                       
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import misc

plt.style.use('ggplot')

#define functions here
#normalize the features
def featureNormalize(X):
    m,n = X.shape
    mu = np.zeros(n)
    sigma =np.zeros(n)
    for i in np.arange(n):
        mean = X[:,i].mean()
        std = X[:,i].std()
        mu[i] = mean
        sigma[i] = std
        X[:,i] = (X[:,i]-mean)/std
    return X, mu, sigma

def pca(X):
    m,n = X.shape
    sig = (1./m)*X.T.dot(X)
    U,S,v = np.linalg.svd(sig)
    print(U)
    print(S)
    print('U shape', U.shape,' S shape ',S.shape)
    return U, S

def projectData(X, U, K):
    U_reduce = U[:,:K]
    Z = X.dot(U_reduce)
    return Z    

def recoverData(Z, U, K):
    U_reduce = U[:,:K]
    X_approx = Z.dot(U_reduce.T)
    return X_approx

#display the image data sample                                                                                                              
def displayData(X, example_width):
    m,n = X.shape
    if(example_width<1):
        example_width = int(np.round(np.sqrt(n)))
    example_height = int(n/example_width)
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))
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
            display_array[pad_x:(pad_x+example_height),pad_y:(pad_y+example_width)]=np.reshape(X[curr_ex,:],(example_height,example_width))\
/max_val
            curr_ex += 1
        if curr_ex>m:
            break

    plt.imshow(display_array,extent=[0,10,0,10],cmap=plt.cm.Greys_r)
    
def calculateVariance(S,K):
    n = S.size
    print('S dims: ',S.shape)
    total_var = S.sum()
    K_var = S[:K].sum()
    print('{:.1f}% of variance is retained '.format(K_var/total_var*100.))
    
    plt.figure()
    loop_K = np.arange(n)
    loop_var = np.zeros(n)
    for i in loop_K:
        loop_var[i] = S[:i].sum()/total_var*100.
        print('K = {}, variance {:.2f}%'.format(i+1,loop_var[i]))
    plt.plot(loop_K,loop_var,'ro')
    plt.title('Variance of PCA')
    plt.xlabel('number of principal components K')
    plt.ylabel('PCA variance (%)')    
    

#main execute code
def main():
    path='/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex7/ex7/'
    filename1 = path+'ex7data1.mat'
    train_data1 = loadmat(filename1)
    print(train_data1.keys())
    X = train_data1['X']
    print(X.shape)
    print('feature 0, min {:.2f}, max {:.2f} ;\n feature 1, min {:.2f}, max {:.2f} '.format(X.min(axis=0)[0],X.max(axis=0)[0],X.min(axis=0)[1],X.max(axis=0)[1]))
    #visualize data
    plt.figure(1)
    plt.scatter(X[:,0],X[:,1],c='b',marker='o',s=30)
    plt.xlim(0.5,6.5)
    plt.ylim(2,8)
    
    #scale the feature and run the PCA method
    X_norm,mu,sigma = featureNormalize(X)
    print("mu ",mu,' sigma ',sigma)
    U,S = pca(X_norm)
    calculateVariance(S,1)
    #visualize the PCA procedure
    plt.figure(3)
    plt.scatter(X_norm[:,0],X_norm[:,1],c='b',marker='o',s=30)
    plt.plot(mu,mu+1.5*S[0]*U[:,0],'k-',linewidth=2)
    plt.plot(mu,mu+1.5*S[1]*U[:,1],'k-',linewidth=2)
    
    #dimension reduction
    K = 1
    Z = projectData(X_norm, U, K)
    print(Z[0])
    X_rec  = recoverData(Z, U, K)
    print(X_rec[0])
    plt.figure(4)
    plt.scatter(X_norm[:,0],X_norm[:,1],c='b',marker='o',s=30,label='X_norm')
    plt.plot(X_rec[:,0],X_rec[:,1],'ro',label='X_projection')
    for i in np.arange(X_norm.shape[0]):
        plt.plot([X_norm[i,0],X_rec[i,0]],[X_norm[i,1],X_rec[i,1]],'--k',linewidth=1)
    
    plt.xlim(-4,3)
    plt.ylim(-4,3)
    plt.title('Visualize the Projections')
    plt.legend(loc='upper left')
    
    #load image and reduce dims on the image
    image_file = path+'ex7faces.mat'
    train_data2 = loadmat(image_file)
    print(train_data2.keys())
    im_X = train_data2['X']
    print('image X shape ',im_X.shape)
    
    plt.figure(6)
    displayData(im_X[:100,:],0)
    
    im_X_norm, im_mu, im_sigma = featureNormalize(im_X)
    im_U,im_S = pca(im_X_norm)
    
    plt.figure(7)
    displayData(U[:,:36].T,0)
    
    im_K = 100;
    im_Z = projectData(im_X_norm, im_U, im_K)
    im_X_rec  = recoverData(im_Z, im_U, im_K)
    calculateVariance(im_S,im_K)
    plt.figure(9)
    displayData(im_X_norm[:100,:],0)
    plt.title('Original faces')
    
    plt.figure(10)
    displayData(im_X_rec[:100,:],0)
    plt.title('reconstructed faces')

    plt.show()

if __name__=='__main__':
    main()
