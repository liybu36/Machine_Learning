'''
Stanford Machine Learning Course, ex8
Movies Rating,collaborative filtering, low rank matrix factorization 
implemented by Hao Qian, Apr 7, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import fmin_cg
import re

plt.style.use('ggplot')

#function definition
#COFICOSTFUNC Collaborative filtering cost function
def cofiCostFunc(params, Y, R, num_users, num_movies,num_features, la):
    '''
    X = params[:num_movies*num_features].reshape((num_movies,num_features))
    Theta = params[num_movies*num_features:].reshape((num_users,num_features))
    predict_y = np.multiply(X.dot(Theta.T),R)
    diff = predict_y - Y
    '''
    X, Theta = unroll_params(params, num_users, num_movies,num_features)
    diff = calculatediff(X, Theta, Y, R) 
    sum = np.power(diff,2).sum()
    X_sum = (la/2.)*(np.power(X*1.0,2).sum())
    Theta_sum = (la/2.)*(np.power(Theta*1.0,2).sum())
    J = 0.5*sum + X_sum + Theta_sum
    return J
    
def unroll_params(params, num_users, num_movies,num_features):
    X = params[:num_movies*num_features].reshape((num_movies,num_features))
    Theta = params[num_movies*num_features:].reshape((num_users,num_features))    
    return X, Theta
    
def calculatediff(X,Theta, Y, R):
    predict_y = np.multiply(X.dot(Theta.T),R)
    diff = predict_y - Y
    return diff    
    
def cofiXgrad(X,Theta,diff,la):
    #X_grad is (num_movies,num_features) ndims
    X_grad = diff.dot(Theta) + la*X
    return X_grad

def cofiThetagrad(X,Theta,diff,la):
    #Theta_grad is (num_users,num_features) ndims
    Theta_grad = diff.T.dot(X) + la*Theta
    return Theta_grad

def cofigrad(params,Y,R,num_users,num_movies,num_features,la):
    X, Theta = unroll_params(params, num_users, num_movies,num_features)
    diff = calculatediff(X, Theta, Y, R) 
    X_grad = cofiXgrad(X, Theta, diff, la)
    Theta_grad = cofiThetagrad(X, Theta, diff, la)
    grad = np.append(X_grad.ravel(),Theta_grad.ravel(),axis=0)
    return grad

#check the gradient
def checkCostFunction(la=0):
    test_users = 5
    test_movies = 4
    test_features = 3
    X_t = np.random.randn(test_movies,test_features)
    Theta_t = np.random.randn(test_users,test_features)
    Y = X_t.dot(Theta_t.T)
    Y[np.random.randn(Y.shape[0],Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y!=0]=1
    
    X = np.random.randn(X_t.shape[0],X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0],Theta_t.shape[1])
    #calculate cofi gradient and cost
    cost = cofiCostFunc(np.append(X.ravel(),Theta.ravel(),0),Y,R,test_users, test_movies,test_features,la)
    grad = cofigrad(np.append(X.ravel(),Theta.ravel(),0),Y,R,test_users, test_movies,test_features,la)
    X_grad,Theta_grad = unroll_params(grad, test_users, test_movies, test_features)
    #calculate numerical gradient
    numgrad = computeNumericalGradient(np.append(X.ravel(),Theta.ravel(),0),Y,R,test_users, test_movies,test_features,la)
    print('cofiCost Grad ',X_grad)
    print('Numeric Grad',numgrad)
    diff = np.linalg.norm(numgrad-Theta_grad)/np.linalg.norm(numgrad+Theta_grad)
    print('difference in grad of Theta',diff)

def computeNumericalGradient(params, Y, R, num_users, num_movies,num_features, la):
    X, Theta = unroll_params(params, num_users, num_movies,num_features)
    numgrad = np.zeros(Theta.shape)
    perturb = np.zeros(Theta.shape)
    e = 1e-4
    for i in np.arange(Theta.shape[0]):
        for j in np.arange(Theta.shape[1]):
            perturb[i,j]=e
            J1 = cofiCostFunc(np.append(X.ravel(),(Theta-perturb).ravel(),0), Y, R, num_users, num_movies, num_features, la)
            J2 = cofiCostFunc(np.append(X.ravel(),(Theta+perturb).ravel(),0), Y, R, num_users, num_movies, num_features, la)
            numgrad[i,j] = (J2-J1)/(2*e)
            perturb[i,j]=0
    return numgrad

#load movie list
def loadMovieList(file):
    with open(file,'r',encoding="ISO-8859-1") as f:
        data = f.readlines()
        
    print(len(data))
    n = len(data)
    movieList = []
    for i in np.arange(n):
        ss = re.search('\s+(.*)',data[i])
        movieList.append(ss.group(1))

    print(movieList)
    return movieList    

def normalizeRatings(Y, R):
    Y_norm = np.zeros(Y.shape)
    Y_mean = np.zeros(Y.shape[0])
    m = Y.shape[0]
    for i in np.arange(m):
        cut = R[i]>0
        Y_mean[i] = Y[i,cut].mean()
        Y_norm[i,cut] = Y[i,cut] - Y_mean[i]
        
    print('Y_norm \n',Y_norm)
    print('Y_mean \n',Y_mean)    
    return Y_norm,Y_mean    
    

#main part to execute the code
def main():
    path='/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex8/ex8/'
    file1 = path+'ex8_movies.mat'
    train_data1 = loadmat(file1)
    print(train_data1.keys())
    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
    Y = train_data1['Y']
    print('rating max {} and min {}'.format(Y.max(),Y.min()))
    #R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
    R = train_data1['R']
    print(Y.shape,' Y and R shape ',R.shape)
    #print(Y)
    #print(R)
    print('Average rating for mive 1 (Toy Story): {:.2f}'.format(Y[0,:].mean()))
    
    #visualie user rating
    plt.figure(1)
    plt.imshow(Y,cmap=plt.cm.magma)
    plt.colorbar()
    plt.ylabel('Movies')
    plt.xlabel('Users')
    
    #load pretrained data
    file2 = path+'ex8_movieParams.mat'
    train_data2 = loadmat(file2)
    print(train_data2.keys())
    num_users = train_data2['num_users'][0,0]
    num_movies=train_data2['num_movies'][0,0]
    num_features=train_data2['num_features'][0,0]
    Theta = train_data2['Theta']
    X = train_data2['X']
    print('num_users: ',num_users,' num_movies ',num_movies,' num_features ',num_features)
    print(X.shape,' X shape, Theta shape',Theta.shape)
    
    #Test: Evaluate cost function
    test_users = 4
    test_movies = 5
    test_features = 3
    test_X = X[:test_movies,:test_features]
    test_Theta = Theta[:test_users,:test_features]
    test_Y = Y[:test_movies,:test_users]
    test_R = R[:test_movies,:test_users]
    test_J = cofiCostFunc(np.append(test_X.ravel(),test_Theta.ravel(),0), test_Y, test_R, test_users, test_movies,test_features, 0)
    print('test J is {:.2f} '.format(test_J))
    checkCostFunction(0)
    test_J1 = cofiCostFunc(np.append(test_X.ravel(),test_Theta.ravel(),0), test_Y, test_R, test_users, test_movies,test_features, 1.5)
    print('test J1 is {:.2f} '.format(test_J1))
    checkCostFunction(1.5)
    
    #load movie list
    file3 = path+'movie_ids.txt'
    movielist = loadMovieList(file3)
    n = len(movielist)
    my_ratings = np.zeros(n)
    my_ratings[0] = 4
    my_ratings[97] = 2
    my_ratings[6] = 3
    my_ratings[11] = 5
    my_ratings[53] = 4
    my_ratings[63] = 5
    my_ratings[65] = 3
    my_ratings[68] = 5
    my_ratings[182] = 4
    my_ratings[225] = 5
    my_ratings[354] = 5
    for i in np.arange(n):
        if my_ratings[i]>0:
            print('Moive {} with rating {}'.format(movielist[i],my_ratings[i]))
    
    #learn moive rating        
    my_ratings = my_ratings.reshape((n,1))
    Y1 = np.append(Y,my_ratings,axis=1)
    my_R = np.zeros((n,1))
    my_R[my_ratings>0] = 1
    R1 = np.append(R,my_R,axis=1)
    
    #normalize ratings
    Ynorm, Ymean = normalizeRatings(Y1, R1)
    norm_movies, norm_users = Ynorm.shape
    norm_features = num_features
    print(norm_movies, norm_users,norm_features)
    norm_X = np.random.randn(norm_movies,norm_features)
    norm_Theta = np.random.randn(norm_users,norm_features)
    norm_params = np.append(norm_X.ravel(),norm_Theta.ravel(),0)
    norm_la = 10
    xopt = fmin_cg(cofiCostFunc,norm_params,fprime=cofigrad,args=(Ynorm,R1,norm_users,norm_movies,norm_features,norm_la),maxiter=100)
    xopt_X, xopt_Theta = unroll_params(xopt, norm_users, norm_movies, norm_features)
    
    #recommendation 
    p = xopt_X.dot(xopt_Theta.T)
    predict_p = p[:,norm_users-1] + Ymean
    predict_idx = np.argsort(predict_p)[::-1]
    predict_p = np.sort(predict_p)[::-1]
    for i in np.arange(10):
        movie = movielist[predict_idx[i]]
        print('Predicting rating {:.1f} for {}'.format(predict_p[i],movie))
    
    
    plt.show()

if __name__=='__main__':
    main()