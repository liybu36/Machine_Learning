'''
Machine_Learning_Stanford ex4
Neural Networks Learning
implemented by Hao Qian, Mar 26, 2017
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import *
from scipy.io import loadmat

plt.style.use('ggplot')

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;

#display the image data sample
def displayData(X, example_width):
    m,n = X.shape
    if(example_width<1):
        example_width = np.round(np.sqrt(n))
    example_height = n/example_width
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m/display_rows)
    example_width = np.int32(example_width)
    example_height = np.int32(example_height)
    display_rows = np.int32(display_rows)
    display_cols = np.int32(display_cols)
    print("example width ",example_width, " height ",example_height)
    print("display row ",display_rows," cols ",display_cols)
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

    plt.imshow(display_array,extent=[0,10,0,10],cmap=plt.cm.gray)

#calculate logistic function
def sigmoid(z):
    g = 1./(1.+np.exp(-z))
    return g

#sigmoid Gradient
def sigmoidGradient(z):
    g_prime = sigmoid(z)*(1-sigmoid(z))
    return g_prime

#unroll the theta1 and theta2 from nn_params
def unroll_theta(nn_params):
    #Theta1 is 25*401 ndims
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    #Theta2 is 10*26 ndims
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    return Theta1, Theta2

#calculate vectorized cost function
def computeCost(nn_params,X,y):
    m = X.shape[0]
    Theta1, Theta2 = unroll_theta(nn_params)
    h = NeuralFeedForward(Theta1,Theta2,X)
    J=(1./m)*(-y.transpose().dot(np.log(h))-(1-y).transpose().dot(np.log(1-h)))
    J_max = J.diagonal() #one sample only has one y value so that is the diagonal part of the matrix
    J = J_max.sum()
    return J

#calculate regularized logistic regression cost function
def computesCostReg(nn_params,X,y,la):
    m=X.shape[0]
    normal_J = computeCost(nn_params,X,y)
    Theta1, Theta2 = unroll_theta(nn_params)
    sum_Theta1 = ((Theta1.dot(Theta1.T)).diagonal() - Theta1[:,0]**2).sum()
    sum_Theta2 = ((Theta2.dot(Theta2.T)).diagonal() - Theta2[:,0]**2).sum()
    reg_J = normal_J + (la/(2*m))*(sum_Theta1 + sum_Theta2)
    return reg_J

def NeuralFeedForward(Theta1,Theta2,X):
    m = X.shape[0]
    K = Theta2.shape[0]
    a2 = sigmoid(Theta1.dot(X.transpose())) #25*m dimensions
    a2_0 = np.ones((1,m))
    a2 = np.append(a2_0,a2,0)
    a3 = sigmoid(Theta2.dot(a2)) #10*m dimensions
    print("a3 shape ",a3.shape)
    return a3
    '''
    h = np.zeros(m)
    for i in np.arange(m):
        for j in np.arange(K):
            if a3[j,i] == a3[:,i].max():
                h[i] = j+1
    h=np.reshape(h,(m,1))
    return h
    '''

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
    print("y map shape ",y_map.shape)
    return y_map

#random initialize theta
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.randn(L_out,L_in+1)*2*epsilon_init - epsilon_init
    print("W shape", W.shape)
    return W

def BackCostFuncGradient(nn_params,X,y):
    #Theta1 is 25*401 ndims
    #Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1))
    #Theta2 is 10*26 ndims
    #Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):],(num_labels,hidden_layer_size+1))
    Theta1, Theta2 = unroll_theta(nn_params)
    m = X.shape[0]#5000 samples
    Theta1_grad = np.zeros(Theta1.shape)#25*401 ndims
    Theta2_grad = np.zeros(Theta2.shape)#10*26 ndims

    z2 = Theta1.dot(X.T)
    a2 = sigmoid(z2) #25*m dimensions
    a2_0 = np.ones((1,m))
    a2_1 = np.append(a2_0,a2,0) #26*m ndims
    z3 = Theta2.dot(a2_1)
    a3 = sigmoid(z3) #10*m dimensions

    del_3 = a3 - y # y is actually y_map, k*m ndims
    #do element wise multipliy with * or np.multiply(a,b)
    del_2 = Theta2.T[1:,].dot(del_3)*sigmoidGradient(z2) # 25*m ndims

    Theta2_grad_temp = (1./m)*(del_3.dot(a2.T)) #K*25 ndims
    Theta1_grad_temp = (1./m)*(del_2.dot(X[:,1:])) #25*400 ndims
    Theta1_grad = np.append(np.zeros((Theta1_grad_temp.shape[0],1)),Theta1_grad_temp,1)
    Theta2_grad = np.append(np.zeros((Theta2_grad_temp.shape[0],1)),Theta2_grad_temp,1)

    grad = np.append(Theta1_grad.ravel(),Theta2_grad.ravel(),0)
    print("grad shape",grad.shape)
    return grad

def BackCostFuncGradientReg(nn_params,X,y,la):
    Theta1, Theta2 = unroll_theta(nn_params)
    nn_params_grad = BackCostFuncGradient(nn_params,X,y)
    Theta1_grad, Theta2_grad = unroll_theta(nn_params_grad)
    m = X.shape[0]
    Theta1_grad[:,1:] += (la/m)*Theta1[:,1:]
    Theta2_grad[:,1:] += (la/m)*Theta2[:,1:]
    grad = np.append(Theta1_grad.ravel(),Theta2_grad.ravel(),0)
    return grad

#Gradient Checking
def computeNumericalGradient(nn_params,X,y):
    epi = 0.0001
    #Theta1, Theta2 = unroll_theta(nn_params)
    size = len(y)
    nn_params_plus = nn_params
    nn_params_minus = nn_params
    numerical_grad = np.zeros(nn_params.shape)
    back_grad = BackCostFuncGradient(nn_params,X,y)
    for i in np.arange(size):
        nn_params_plus[i] += epi
        nn_params_minus[i] -= epi
        J_plus = computeCost(nn_params_plus,X,y)
        J_minus = computeCost(nn_params_minus,X,y)
        numerical_grad[i] = (J_plus - J_minus)/(2*epi)
        diff = back_grad[i]-numerical_grad[i]
        print("Theta ", i, " difference is ",diff)

#Gradient Checking Regularization
def computeNumericalGradientReg(nn_params,X,y,la):
    epi = 0.0001
    #Theta1, Theta2 = unroll_theta(nn_params)
    size = len(y)
    nn_params_plus = nn_params
    nn_params_minus = nn_params
    numerical_grad = np.zeros(nn_params.shape)
    back_gradreg = BackCostFuncGradientReg(nn_params,X,y,la)
    for i in np.arange(size):
        nn_params_plus[i] += epi
        nn_params_minus[i] -= epi
        J_plus = computesCostReg(nn_params_plus,X,y,la)
        J_minus = computesCostReg(nn_params_minus,X,y,la)
        numerical_grad[i] = (J_plus - J_minus)/(2*epi)
        diff = back_gradreg[i]-numerical_grad[i]
        print("Regularization Theta ", i, " difference is ",diff)

#prefict the traing accuracy
def predict(nn_params,X):
    m = X.shape[0]
    Theta1, Theta2 = unroll_theta(nn_params)
    fig = plt.figure(2)
    displayData(Theta1[:,1:],0)

    K = Theta2.shape[0]
    z2 = Theta1.dot(X.T)
    a2 = sigmoid(z2) #25*m dimensions
    a2_0 = np.ones((1,m))
    a2_1 = np.append(a2_0,a2,0) #26*m ndims
    z3 = Theta2.dot(a2_1)
    a3 = sigmoid(z3) #10*m dimensions
    h = np.zeros(m)
    for i in np.arange(m):
        for j in np.arange(K):
            if a3[j,i] == a3[:,i].max():
                h[i] = j+1
    h=np.reshape(h,(m,1))
    return h

#debug initialize weight
def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out,fan_in+1))
    W = np.sin(np.arange(W.size))/10
    W = np.reshape(W,(fan_out,fan_in+1))
    print("debug W shape ", W.shape)
    return W

def checkNNGradients(la):
    m = 10;
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    nn_params = np.append(Theta1.ravel(),Theta2.ravel(),0)
    X  = debugInitializeWeights(m, input_layer_size - 1)
    print("X shape ",X.shape)
    X = np.append(np.ones((m,1)),X,1)
    y  = 1 + np.mod(np.arange(m), num_labels)
    print(y.shape)
    y = mapy(y)
    #grad = BackCostFuncGradient(nn_params,X,y)
    computeNumericalGradientReg(nn_params,X,y,la)


#main function to execute the MultiClass and Neural Networks
path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex4/ex4/"
filename=path+"ex4data1.mat"
#load pixels into data as dict
train_data = loadmat(filename)
print(train_data.keys())
X=train_data['X']
y=train_data['y']
print("X shape ",X.shape," y shape ",y.shape)
m,n = X.shape

#random select a list from X and display the handwritten
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

#Neural Networks
weightfilename = path+'ex4weights.mat'
weight=loadmat(weightfilename)
print(weight.keys())
Theta1 = weight['Theta1']
Theta2 = weight['Theta2']
print("Theta1 shape ",Theta1.shape)
print("Theta2 shape ",Theta2.shape)

nn_params_test = np.append(Theta1.ravel(),Theta2.ravel(),0)

cost = computeCost(nn_params_test,X2,y_map)
print("cost ",cost)
la=1
costreg = computesCostReg(nn_params_test,X2,y_map,la)
print("costreg ",costreg)

#apply backpropagation
#L_out and L_in are outgoing units and incoming units
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
print("initial_Theta1 ",initial_Theta1)
print("initial_Theta2 ",initial_Theta2)

nn_params = np.append(initial_Theta1.ravel(),initial_Theta2.ravel(),0)
print("nn_params shape ", nn_params.shape)
test_g_gradient =sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
print("test_g_gradient ",test_g_gradient)

#Gradient Checking
computeNumericalGradient(nn_params,X2,y_map)
computeNumericalGradientReg(nn_params,X2,y_map,la)

costreg = computesCostReg(nn_params_test,X2,y_map,3)
print("la 3 => costreg ",costreg)
checkNNGradients(3)

#minimization
xopt = fmin_cg(computeCost,nn_params,fprime=BackCostFuncGradient,args=(X2,y_map),maxiter=50)
print("xopt ",xopt)
xopt_a3 = predict(xopt,X2)
print('Neural Network Train Accuracy: %f' % ((y[(xopt_a3 == y)].size / float(y.size)) * 100.0))

#as la=0.05 and maxiter=400 the Accuracy reaches 100%, may cause overfitting issue
#xoptreg = fmin_cg(computesCostReg,nn_params,fprime=BackCostFuncGradientReg,args=(X2,y_map,0.05),maxiter=400)
xoptreg = fmin_cg(computesCostReg,nn_params,fprime=BackCostFuncGradientReg,args=(X2,y_map,la),maxiter=50)
print("xopt reg ",xoptreg)
xoptreg_a3 = predict(xoptreg,X2)
print('Neural Network Train Accuracy: %f' % ((y[(xoptreg_a3 == y)].size / float(y.size)) * 100.0))

plt.show()
