from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np
from pathlib import Path

TANK_location = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200928-111539')
X_location = next(TANK_location.glob('*regressionData_X.csv'))
y_location = next(TANK_location.glob('*regressionData_y.csv'))


X = np.loadtxt(str(X_location), delimiter=',')
y = np.loadtxt(str(y_location), delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

reg1 = MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, learning_rate_init=0.01)
reg2 = MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, learning_rate_init=0.01)

reg1.fit(X_train,y_train[:,0])
reg2.fit(X_train,y_train[:,1])

y_test_fake = y_test.copy()
np.random.shuffle(y_test_fake)

def rmse(y_true, y_pred, std=1):
    return np.mean(((y_true - y_pred) ** 2)) ** .5

error_x = rmse(y_test[:,0], reg1.predict(X_test))
error_y = rmse(y_test[:,1], reg2.predict(X_test))
error_x_fake = rmse(y_test_fake[:,0], reg1.predict(X_test))
error_y_fake = rmse(y_test_fake[:,1], reg2.predict(X_test))


print('error x : %.3f' % (error_x))
print('error y : %.3f' % (error_y))
print('error x_fake : %.3f' % (error_x_fake))
print('error y_fake : %.3f' % (error_y_fake))



plt.clf()
plt.plot(y_test[:,0],'g')
plt.plot(reg1.predict(X_test),'r')




r1 = 600
r2 = 700

y_predict = np.vstack((reg1.predict(X), reg2.predict(X))).T

fig1 = plt.figure(1,figsize=(12, 3))
fig1.clf()
ax1 = fig1.add_subplot(1,1,1)
fig1.suptitle('X-axis')
ax1.plot(y[r1:r2,0],color='#00C000')
ax1.plot(y_predict[r1:r2,0],color='#FF0066')
ax1.legend(['Real','Predicted'])
ax1.set_ylabel('Location (pixel)')
ax1.set_xlabel('Time (s)')

fig2 = plt.figure(2,figsize=(12, 3))
fig2.clf()
ax2 = fig2.add_subplot(1,1,1)
fig2.suptitle('Y-axis')
ax2.plot(y[r1:r2,1],color='#00C000')
ax2.plot(y_predict[r1:r2,1],color='#FF0066')
ax2.legend(['Real','Predicted'])
ax2.set_ylabel('Location (pixel)')
ax2.set_xlabel('Time (s)')

fig3 = plt.figure(3,figsize=(6, 6))
fig3.clf()
ax3 = fig3.add_subplot(1,1,1)
fig3.suptitle('2D plot')
ax3.plot(y[400:500,0], y[400:500,1],color='#00C000')
ax3.plot(y_predict[400:500,0], y_predict[400:500,1],color='#FF0066')
ax3.legend(['Real','Predicted'])

diff_y = y - y_predict

np.mean((diff_y/np.std(y,0))**2,0)

with open('test.csv','w') as f:
    wtr = csv.writer(f, delimiter='\t')
    for line in y:
        wtr.writerow(line)
