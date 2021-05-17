from scipy.io import loadmat
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = loadmat(r'C:\Users\Knowblesse\Downloads\ILensemble_location.mat')

X = data.get('X')
Y = data.get('Y')

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)

reg = MLPRegressor(hidden_layer_sizes=100,warm_start=True, max_iter=1000, learning_rate_init=0.05)

reg.fit(X_train,Y_train)
Y_predict = reg.predict(X)


fig1 = plt.figure(1,figsize=(12, 3))
fig1.clf()
ax1 = fig1.add_subplot(1,1,1)
fig1.suptitle('X-axis')
ax1.plot(Y[300:500,0],color='#00C000')
ax1.plot(Y_predict[300:500,0],color='#FF0066')
ax1.legend(['Real','Predicted'])
ax1.set_ylabel('Location (pixel)')
ax1.set_xlabel('Time (s)')

fig2 = plt.figure(2,figsize=(12, 3))
fig2.clf()
ax2 = fig2.add_subplot(1,1,1)
fig2.suptitle('Y-axis')
ax2.plot(Y[300:500,1],color='#00C000')
ax2.plot(Y_predict[300:500,1],color='#FF0066')
ax2.legend(['Real','Predicted'])
ax2.set_ylabel('Location (pixel)')
ax2.set_xlabel('Time (s)')

fig3 = plt.figure(3,figsize=(6, 6))
fig3.clf()
ax3 = fig3.add_subplot(1,1,1)
fig3.suptitle('2D plot')
ax3.plot(Y[400:500,0], Y[400:500,1],color='#00C000')
ax3.plot(Y_predict[400:500,0], Y_predict[400:500,1],color='#FF0066')
ax3.legend(['Real','Predicted'])