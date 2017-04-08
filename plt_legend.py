import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm
from  mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

fig = plt.figure(1)
red_path = mpatches.Patch(color='r',label='The red data')
plt.legend(handles=[red_path])

fig = plt.figure(2)
blue_lines = mlines.Line2D(np.arange(10),np.arange(10),color='blue',marker='*',markersize=15,label='Blue Stars')
plt.legend(handles=[blue_lines])

fig = plt.figure(3)
plt.subplot(211)
plt.plot([1,2,3],label='test1')
plt.plot([3,2,1],label='test2')
plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3,ncol=2,mode='expand',borderaxespad=0.)

plt.subplot(223)
plt.plot([1,2,3], label="test1")
plt.plot([3,2,1], label="test2")
# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

fig = plt.figure(4)
line1, = plt.plot([1,2,3],c='b',linestyle='--',label='line 1')
line2, = plt.plot([3,2,1],label='line 2',linewidth=4)
first_legend = plt.legend(handles=[line1],loc=1)
ax = plt.gca().add_artist(first_legend)
plt.legend(handles=[line2],loc=4)

fig = plt.figure(5)
line1, = plt.plot([3,2,1], marker='o', label='Line 1')
line2, = plt.plot([1,2,3], marker='o', label='Line 2')
plt.legend(loc='center right')

fig = plt.figure(6)
z = np.random.randn(10)
red_dot, = plt.plot(z,'ro',markersize=15)
white_cross, = plt.plot(np.arange(10),'w+',markeredgewidth=3,markersize=15)
plt.legend([red_dot,white_cross],['Attr A','Attr A+B'])
plt.axis([-1,1,-1,20])
#plt.axis(v)  v = [xmin, xmax, ymin, ymax]

fig = plt.figure(7)
t = np.arange(0.,5.,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

fig = plt.figure(8)
line, = plt.plot(np.arange(5),'-')
#line.set_antialiased(False)

fig = plt.figure(9)
lines = plt.plot(np.arange(5),np.arange(5)+1,np.arange(5),np.arange(5)+2)
plt.setp(lines, color='r', linewidth=2.0)

def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1 = np.arange(0,5.,0.1)
t2 = np.arange(0,5.,0.02)
plt.figure(10)
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')

plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),'r--')

plt.figure(11)
np.random.seed(19999)
mu, sigma = 100,15
x = mu + sigma*np.random.randn(10000)
n,bins,patches = plt.hist(x,50,normed=1,facecolor='g',alpha=0.95)
plt.xlabel('Smarts',fontsize=14, color='red')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.figure(12)
plt.plot(np.arange(1000))
plt.yscale('log') # linear, symlog, logit

# create some data to use for the plot
plt.figure(13)
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000]/0.05)               # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)]*dt  # colored noise

# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Gaussian colored noise')

# this is an inset axes over the main axes
a = plt.axes([.65, .6, .2, .2])
n, bins, patches = plt.hist(s, 400, normed=1)
plt.title('Probability')
plt.xticks([])
plt.yticks([])

# this is another inset axes over the main axes
a = plt.axes([0.2, 0.6, .2, .2]) #, facecolor='y')
plt.plot(t[:len(r)], r)
plt.title('Impulse response')
plt.xlim(0, 0.2)
plt.xticks([])
plt.yticks([])

#share x axis
x = np.linspace(0,2*np.pi,400)
y = np.sin(x**2)
f,ax = plt.subplots(2,sharex=True)
ax[0].plot(x,y)
ax[0].set_title('Shareing X axis')
ax[1].scatter(x,y)

fig = plt.figure(14)
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

fig = plt.figure(15)
ax = fig.add_subplot(111, projection='3d')
# Grab some test data.
#X, Y, Z = Axes3D.get_test_data(0.05)
X,Y,Z = x,y,z
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

fig = plt.figure(16)
ax = fig.gca(projection='3d')
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)

#sigmoid and tanh
def sigmoid(z):
    return 1./(1.+np.exp(-z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

z = np.arange(-5,5,0.5)
plt.figure(17)
plt.plot(z,sigmoid(z),'r--',label='sigmoid',linewidth=2)
plt.plot(z,tanh(z),'b',label='tanh',linewidth=2)
plt.xlabel('z')
plt.ylabel('f(z) function')
plt.legend(loc='center right')

plt.show()
