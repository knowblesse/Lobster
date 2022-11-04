from scipy.io import loadmat
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# from  LocationRegression.LocationRegressor.LocationRegressor import loadData
from pathlib import Path
import os


def loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData=False):
    # Check if the video file is buttered
    butter_location = [p for p in tankPath.glob('*_buttered.csv')]

    if len(butter_location) == 0:
        raise (BaseException("Can not find a butter file in the current Tank location"))
    elif len(butter_location) > 1:
        raise (BaseException("There are multiple files ending with _buttered.csv"))

    # Check if the neural data file is present
    wholeSessionUnitData_location = [p for p in tankPath.glob('*_wholeSessionUnitData.csv')]

    if len(wholeSessionUnitData_location) == 0:
        raise (BaseException("Can not find a regression data file in the current Tank location"))
    elif len(wholeSessionUnitData_location) > 1:
        raise (BaseException("There are multiple files ending with _wholeSessionUnitData.csv"))

    # Check Video FPS
    fpsFileName = tankPath / 'FPS.txt'
    video_frame_rate = int(np.loadtxt(fpsFileName))

    # Load file
    butter_data = np.loadtxt(str(butter_location[0]), delimiter='\t')
    neural_data = np.loadtxt(str(wholeSessionUnitData_location[0]), delimiter=',')

    # Check if -1 value exist in the butter data
    if np.any(butter_data == -1):
        raise (BaseException("-1 exist in the butter data. check with the relabeler"))

    # Generate Interpolation function
    intp_r = interp1d(butter_data[:, 0], butter_data[:, 1], kind='linear')
    intp_c = interp1d(butter_data[:, 0], butter_data[:, 2], kind='linear')

    # Find midpoint of each neural data
    #   > If neural data is collected from 0 ~ 0.5 sec, (neural_data_rate=2), then the mid-point of the
    #       neural data is 0.25 sec. The next neural data, which is collected during 0.5~1.0 sec, has
    #       the mid-point of 0.75 sec.
    midPointTimes = truncatedTime_s + (1 / neural_data_rate) * np.arange(neural_data.shape[0]) + 0.5 * (
                1 / neural_data_rate)

    y_r = intp_r(midPointTimes * video_frame_rate)
    y_c = intp_c(midPointTimes * video_frame_rate)

    # If removeNestingData is set True, remove all points which has the column value smaller than 200
    if removeNestingData:
        print('removing nesting')
        neural_data = neural_data[y_c >= 200, :]
        y_r = y_r[y_c >= 200]
        y_c = y_c[y_c >= 200]

    return(neural_data, np.expand_dims(y_r, 1), np.expand_dims(y_c, 1))

tankName = '#21AUG3-211028-165958_PL'
locationDataPath = Path(r"D:/Data/Lobster/LocationRegressionData") / Path(tankName)
behaviorDataPath = Path(r"D:/Data/Lobster/BehaviorData") / Path(tankName).with_suffix('.mat')
neural_data, y_r, y_c = loadData(locationDataPath, neural_data_rate=2, truncatedTime_s=10, removeNestingData=False)
behavior_data = loadmat(behaviorDataPath)

# Parse behavior Data
behavior_data['ParsedData']

