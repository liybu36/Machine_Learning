'''
Stanford Machine Learning Class, Ex6 Spam classification
implemented by Hao Qian, Apr 3, 2017
'''
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import re
from sklearn.svm import LinearSVC, SVC

plt.style.use('ggplot')

path="/Users/hqian/Box Sync/CS/python/python_code/Machine_Learning_Stanford/machine-learning-ex6/ex6/"

#function parts
#read text files
def LoadText(file,option):
    with open(file,'r') as f:
        if option == 0:
            data_file = f.readlines()
        else:
            data_file = f.read().replace('\n','')
    print(data_file)
    return data_file

def getVocabList():
    file = path+'vocab.txt'
    vocab_data = LoadText(file,0)
    '''
    for i in np.arange(len(vocab_data)):
        vocab_data[i] = vocab_data[i].rstrip()
        vocab_data[i] = re.search('\t([a-zA-Z]*)',vocab_data[i]).group(1)
    print(vocab_data)
    print(len(vocab_data))
    return vocab_data
'''
    vocab_list = dict()
    for i in np.arange(len(vocab_data)):
        vocab_data[i] = vocab_data[i].rstrip()
        num = re.search('[0-9]+',vocab_data[i]).group()
        num = int(num)
        ss = re.search('\t([a-zA-Z]*)',vocab_data[i]).group(1)
        vocab_list[num]=ss
    print(vocab_list)
    return vocab_list    
          
def cleanEmail(content):
    content = content.lower()
    content = re.sub('[<>]+','',content)
    content = re.sub('[0-9]+','number',content)
    content = re.sub('(http|https)://[^\s]*','httpaddr',content)
    content = re.sub('[^\s]+@[^\s]+','emailaddr',content)
    content = re.sub('[$]+','dollar',content)
    content = re.sub('[^a-zA-Z0-9 ]+','',content)
    content_list = content.rsplit(' ')
    content_list.remove('')
    print(content_list) 
    return content_list        
          
#clean and select useful information
def processEmail(content):
    vocabList = getVocabList()
    content_list = cleanEmail(content)
    word_indices = [0]*len(content_list)
    for i in np.arange(len(content_list)):
        if content_list[i] in vocabList:
           word_indices[i] = vocabList.index(content_list[i])+1
        else:
            best_pos = 1
            word_tag = False
            for j in vocabList.keys():
                ss = re.match(vocabList[j],content_list[i])
                if ss != None:
                    word_tag = True
                    if len(vocabList[j])>=len(vocabList[best_pos]):
                        best_pos=j
            if word_tag:            
                word_indices[i] = best_pos
                content_list[i] = vocabList[best_pos]       
            else:
                word_indices[i] = 0    
    #sort list by length of element
    #ss.sort(key=lambda s:len(s))
    print(content_list)
    print(word_indices)
    print(len(content_list),len(word_indices))  
    return word_indices              

def emailFeatures(word_indices):
    n = 1899
    '''
    ss = set(word_indices)
    print(len(ss))
    word_indices = np.array(word_indices)
    for i in ss:
        x = np.zeros((n, 1))
        for j in np.arange(len(word_indices)):
            if i==word_indices[j] & i != 0:
                x[word_indices[j]] = 1
        print(x)
'''    
    x = np.zeros((n,1))
    for j in np.arange(len(word_indices)):
        if word_indices[j] != 0:
            x[word_indices[j]]  = 1       
    print((x==1).sum())
    return x.T

#main execute code
file1 = path+'emailSample1.txt'
data_file1 = LoadText(file1,1)
word_indices1  = processEmail(data_file1)
features = emailFeatures(word_indices1)

#load train data
train = loadmat(path+'spamTrain.mat')
print(train.keys()) 
X_train = train['X']
y_train = train['y']
y_train = y_train.reshape((y_train.size,))
print(X_train.shape)
print(y_train.shape)
print(X_train)
print(y_train)

test = loadmat(path+'spamTest.mat')
print(test.keys())
X_test = test['Xtest']
y_test = test['ytest']
y_test = y_test.reshape((y_test.size,))

#train Linear SVC
C=0.1
lin_svc = LinearSVC(C=C,loss='hinge')
lin_svc.fit(X_train,y_train)
print('linearSVC train Score: ',lin_svc.score(X_train,y_train))
print('test score: ',lin_svc.score(X_test,y_test))

#test sample and spam emails
files = ['emailSample2.txt','spamSample1.txt','spamSample2.txt']
for i in np.arange(len(files)):
    file2 = path + files[i]
    data_file2 = LoadText(file2,1)
    word_indices2  = processEmail(data_file2)
    X_sample_2 = emailFeatures(word_indices2)
    y_predict = lin_svc.predict(X_sample_2)
    print(files[i],' y prediction ',y_predict)
    
    
    