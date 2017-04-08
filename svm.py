from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

plt.style.use("ggplot")

np.random.seed(6)
X=np.random.randn(200,2)
print(X)
y = X[:,1] > np.absolute(X[:,0])
print(y)
y=np.where(y,1,-1)
print(y)

#play with different kernels in SVC, the results differ a lot
#try 'linear', 'rbf' , 'poly','sigmoid'
model = SVC(kernel='rbf',random_state=0,gamma=0.5,C=10)
model.fit(X,y)
plot_decision_regions(X,y,model,markers=['o','x'],colors='blue,magenta')
plt.show()
