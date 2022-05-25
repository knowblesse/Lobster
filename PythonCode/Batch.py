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
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import cv2 as cv

# Constant
neural_data_rate = 2 # datapoints per sec. This shows how many X data is present per second.
truncatedTime_s = 10 # sec. matlab data delete the first and the last 10 sec of the neural data.

FolderLocation = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1')

for tank in FolderLocation.glob('#*'):
    tank_name = re.search('#.*',str(tank))[0]
    TANK_location = tank.absolute()

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

    # Check Video FPS
    vc = cv.VideoCapture(str(next(TANK_location.glob('*.avi'))))
    video_frame_rate = vc.get(cv.CAP_PROP_FPS)
    if video_frame_rate % 1 != 0:
        raise(BaseException("Video Frame rate is not a integer!"))
    video_frame_rate = int(video_frame_rate)

    # Load file
    butter_data = np.loadtxt(str(butter_location[0]), delimiter='\t')
    neural_data = np.loadtxt(str(wholeSessionUnitData_location[0]), delimiter=',')

    # Check if -1 value exist in the butter data
    if np.any(butter_data == -1):
        raise(BaseException("-1 exist in the butter data. check with the relabeler"))

    # Interpolate butter data
    #   See Butter package's butterUtil/interpolateButterData for detail

    prev_head_direction = butter_data[0, 3]
    degree_offset_value = np.zeros(butter_data.shape[0])
    for i in np.arange(1, butter_data.shape[0]):
        # if the degree change is more than a half rotation, use the smaller rotation value instead.
        if np.abs(butter_data[i, 3] - prev_head_direction) > 180:
            if butter_data[i, 3] > prev_head_direction:
                degree_offset_value[i:] -= 360
            else:
                degree_offset_value[i:] += 360
        prev_head_direction = butter_data[i, 3]

    # Generate Interpolation function
    intp_x = interp1d(butter_data[:, 0], butter_data[:, 1], kind='linear')
    intp_y = interp1d(butter_data[:, 0], butter_data[:, 2], kind='linear')
    intp_d = interp1d(butter_data[:, 0], np.convolve(butter_data[:, 3] + degree_offset_value, np.ones(5), 'same') / 5, kind='linear')

    # Find midpoint of each neural data
    #   > If neural data is collected from 0 ~ 0.5 sec, (neural_data_rate=2), then the mid-point of the
    #       neural data is 0.25 sec. The next neural data, which is collected during 0.5~1.0 sec, has
    #       the mid-point of 0.75 sec.
    midPointTimes = truncatedTime_s + (1/neural_data_rate)*np.arange(neural_data.shape[0]) + 0.5 * (1/neural_data_rate)

    y_x = np.expand_dims(intp_x(midPointTimes * video_frame_rate), 1)
    y_y = np.expand_dims(intp_y(midPointTimes * video_frame_rate), 1)
    y_d = np.expand_dims(intp_d(midPointTimes * video_frame_rate) % 360, 1)

    X = neural_data
    y = np.concatenate((y_x, y_y, y_d), axis=1)

    # Run Test
    error_x_fake = []
    error_y_fake = []
    error_d_fake = []

    error_x = []
    error_y = []
    error_d = []

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(X):
        X_train = X[train_index,:]
        X_test = X[test_index,:]

        y_train = y[train_index,:]
        y_test = y[test_index, :]

        reg1 = MLPRegressor(hidden_layer_sizes=(200, 50), max_iter=4000, solver='sgd', alpha=1e-3,
                            learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)
        reg2 = MLPRegressor(hidden_layer_sizes=(200, 50), max_iter=4000, solver='sgd', alpha=1e-3,
                            learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)
        reg3 = MLPRegressor(hidden_layer_sizes=(200, 50), max_iter=4000, solver='sgd', alpha=1e-3,
                            learning_rate='invscaling', learning_rate_init=1e-4, power_t=0.2)

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

        error_x_fake.append(rmse(y_test_fake[:, 0], reg1_result))
        error_y_fake.append(rmse(y_test_fake[:, 1], reg2_result))
        error_d_fake.append(rmse(y_test_fake[:, 2], reg3_result))

        error_x.append(rmse(y_test[:,0], reg1_result))
        error_y.append(rmse(y_test[:,1], reg2_result))
        error_d.append(rmse(y_test[:,2], reg3_result))

    print(f'{tank_name} : {np.mean(error_x_fake):.3f}, {np.mean(error_x):.3f}, {np.mean(error_y_fake):.3f}, {np.mean(error_y):.3f}, {np.mean(error_d_fake):.3f}, {np.mean(error_d):.3f}')
