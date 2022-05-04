"""
Location Decoder
@ 2021 Knowblesse
Using the Multi Layer Perceptron based Regressor, decode the current animal's location using the neural data.
Input : 
    TANK_location 
The Tank must have these two data
    1) buttered_data.csv (from butter package)
    2) neural spike data (get it from the Matlab)
"""
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Constant
video_frame_rate = 30 # frames per sec.  notice that this value is different than the buttered data frame rate
neural_data_rate = 2 # datapoints per sec. This shows how many X data is present per second.
truncatedTime_s = 10 # sec. matlab data delete the first and the last 10 sec of the neural data.
# Data Location
TANK_location = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\21JAN2\#21JAN2-210406-190737_IL')

# Check if the video file is buttered
butter_location = [p for p in TANK_location.glob('*_buttered.csv')]

if len(butter_location) == 0:
    raise(BaseException("Can not find a butter file in the current Tank location"))
elif len(butter_location) > 1:
    raise(BaseException("There are multiple files ending with _buttered.csv"))

# Check if the neural data file is present
wholeSessionUnitData_location = [p for p in TANK_location.glob('*_wholeSessionUnitData.csv')]

if len(wholeSessionUnitData_location) == 0:
    raise(BaseException("Can not find a regression data file in the current Tank location"))
elif len(wholeSessionUnitData_location) > 1:
    raise(BaseException("There are multiple files ending with _wholeSessionUnitData.csv"))

# Load file
butter_data = np.loadtxt(str(butter_location[0]), delimiter='\t')
neural_data = np.loadtxt(str(wholeSessionUnitData_location[0]), delimiter=',')

# Check if -1 value exist in the butter data
if np.any(butter_data == -1):
    raise(BaseException("-1 exist in the butter data. check with the relabeler"))

# Find how many butter datapoints correspond to a neural datapoint
butter_data_rate = int(butter_data[1][0]) # frames per datapoint
if video_frame_rate % (butter_data_rate*neural_data_rate) != 0:
    raise(BaseException('video_frame_rate is not dividable by (butter_fps * neural_data_rate)!'))
butter_per_neural = int(video_frame_rate / neural_data_rate / butter_data_rate) # Defines how many datapoints of butter data corresponds to a single neural datapoint

# Remove the first video clip (this is also applied to the neural data from Matlab) and match the size with the neural data
#   cf. since the neural data starts from the 30min long empty array, the size of the neural data is fixed
#       unless the actual data is shorter than the 30minutes.
#       However, butter data gets all the frames from the video, so usually it is longer than the neural data
#       If, presumably the video's frame rate is stable and accurate, then truncating the first <truncatedTime_s> sec
#       from the data and equalizing two dataset will do the job.
numFrames2Delete = int(truncatedTime_s *  (video_frame_rate / butter_data_rate))
butter_data = butter_data[numFrames2Delete: numFrames2Delete + int(neural_data.shape[0]/neural_data_rate * video_frame_rate / butter_data_rate),:]

# Truncate extra data
butter_data = butter_data[0:int(butter_data.shape[0]/butter_per_neural)*butter_per_neural,:]

# Mean multiple butter datapoints to match to a neural datapoint
butter_data_x = butter_data[:,1]
butter_data_y = butter_data[:,2]
butter_data_t = butter_data[:,3]

y_x = np.expand_dims(np.mean(np.reshape(butter_data_x, (-1, butter_per_neural)),axis=1),1)
y_y = np.expand_dims(np.mean(np.reshape(butter_data_y, (-1, butter_per_neural)),axis=1),1)
y_t = np.expand_dims(np.reshape(butter_data_t, (-1, butter_per_neural))[:,1],1) # You can not simply mean the theta. (ex. average of 350, 0, 10 results 120, not 0)

# # Scale
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()
# scaler_t = StandardScaler()
#
# scaler_x.fit(y_x)
# scaler_y.fit(y_y)
# scaler_t.fit(y_t)
#
# y_x = scaler_x.transform(y_x)
# y_y = scaler_y.transform(y_y)
# y_t = scaler_t.transform(y_t)


y = np.concatenate((y_x, y_y, y_t), axis=1)

# If the video is shorter than 30 min, truncate the neural data
if neural_data.shape[0] > y.shape[0]:
    X = neural_data[0:y.shape[0],:]
else:
    X = neural_data

# Run Test
error_x_fake = []
error_y_fake = []
error_t_fake = []

error_x = []
error_y = []
error_t = []

true_ys = []
y_errors = []

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    X_train = X[train_index,:]
    X_test = X[test_index,:]

    y_train = y[train_index,:]
    y_test = y[test_index, :]

    reg1 = MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, learning_rate_init=0.01)
    reg2 = MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, learning_rate_init=0.01)
    reg3 = MLPRegressor(hidden_layer_sizes=(200,50), max_iter=1000, learning_rate_init=0.01)

    reg1.fit(X_train,y_train[:,0])
    reg2.fit(X_train,y_train[:,1])
    reg3.fit(X_train, y_train[:,2])

    y_test_fake = y_test.copy()
    np.random.shuffle(y_test_fake)

    def rmse(y_true, y_pred, std=1):
        return np.mean(((y_true - y_pred) ** 2)) ** .5

    reg1_result = reg1.predict(X_test)
    reg2_result = reg2.predict(X_test)
    reg3_result = reg3.predict(X_test)

    y_errors.append(y_test[:, 1] - reg2_result)
    true_ys.append(y_test[:,1])

    error_x_fake.append(rmse(y_test_fake[:, 0], reg1_result))
    error_y_fake.append(rmse(y_test_fake[:, 1], reg2_result))
    error_t_fake.append(rmse(y_test_fake[:, 2], reg3_result))

    error_x.append(rmse(y_test[:,0], reg1_result))
    error_y.append(rmse(y_test[:,1], reg2_result))
    error_t.append(rmse(y_test[:,2], reg3_result))

    print(f'{error_x_fake[-1]:.3f}, {error_x[-1]:.3f}, {error_y_fake[-1]:.3f}, {error_y[-1]:.3f}, {error_t_fake[-1]:.3f}, {error_t[-1]:.3f}')

print('Copy : ')
print(f'{np.mean(error_x_fake):.3f}, {np.mean(error_x):.3f}, {np.mean(error_y_fake):.3f}, {np.mean(error_y):.3f}, {np.mean(error_t_fake):.3f}, {np.mean(error_t):.3f}')
print('Results : ')
print(f'{np.mean(error_x_fake):.3f}, {np.mean(error_y_fake):.3f}, {np.mean(error_t_fake):.3f}')
print(f'{np.mean(error_x):.3f}, {np.mean(error_y):.3f}, {np.mean(error_t):.3f}')