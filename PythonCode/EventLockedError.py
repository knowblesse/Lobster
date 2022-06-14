"""
EventLockedError
@ 2022 Knowblesse
Generate Event Locked Error plot.
By using all session files, see if the location decoding error changes around the onset of a behavior event.
Files it uses:
    Tank (for video)
    Regression Result (=location decoding result)
    Behavior Data (matlab .m file. has two variables 'ParsedData' and 'behaviorResult'
"""
from scipy.io import loadmat
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

TankParent_Path = Path(r'F:\LobsterData')
RegressionParent_Path = Path(r'F:\Output')
BehaviorParent_Path = Path(r'F:\Output\BehaviorData')
tanks = [f for f in TankParent_Path.glob('#*')]
points = 40
Truncated_time = 10  # sec
Neural_data_rate = 2  # datapoints per sec. This shows how many X data is present per second.

# Data Collector
plot_data_r = np.empty((0,points))
plot_data_r_A = np.empty((0,points))
plot_data_r_E = np.empty((0,points))

plot_data_c = np.empty((0,points))
plot_data_c_A = np.empty((0,points))
plot_data_c_E = np.empty((0,points))

# Run through all session
for tank in tqdm(tanks):
    Session_Name = re.search('#.*',str(tank))[0]
    Video_Path = [f for f in tank.glob('*.avi')]
    if not Video_Path:
        Video_Path = next(tank.glob('*.mp4'))
    else:
        Video_Path = Video_Path[0]
    RegressionResult_Path = next(RegressionParent_Path.glob(Session_Name + '*'))
    matlabFile = loadmat(str(next(BehaviorParent_Path.glob(Session_Name + '*'))))

    # Behavior Data Setup
    behav_result = matlabFile['behaviorResult']
    behav_data = matlabFile['ParsedData']

    numTrial = behav_data.shape[0]

    # Video Setup
    vc = cv.VideoCapture(str(Video_Path.absolute()))
    fps = vc.get(cv.CAP_PROP_FPS)

    # RegressionResult Setup
    reg_dat = np.loadtxt(str(RegressionResult_Path), delimiter=',')

    intp_r_true = interp1d(np.array(np.arange(
        Truncated_time + (0.5 * 1 / Neural_data_rate),
        Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate,
        1 / Neural_data_rate)),
        reg_dat[:, 0],
        kind='linear')
    intp_c_true = interp1d(np.array(np.arange(
        Truncated_time + (0.5 * 1 / Neural_data_rate),
        # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
        Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate,
        1 / Neural_data_rate)),
        reg_dat[:, 1],
        kind='linear')
    intp_r_pred = interp1d(np.array(np.arange(
        Truncated_time + (0.5 * 1 / Neural_data_rate),
        # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
        Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate,
        1 / Neural_data_rate)),
        reg_dat[:, 6],
        kind='linear')
    intp_c_pred = interp1d(np.array(np.arange(
        Truncated_time + (0.5 * 1 / Neural_data_rate),
        # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
        Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate,
        1 / Neural_data_rate)),
        reg_dat[:, 7],
        kind='linear')

    # Generate Lick Onsets
    for i in range(numTrial):
        lickOnset = behav_data[i,0][0,0] + behav_data[i,2][0,0]
        timeArray = np.linspace(lickOnset-5, lickOnset+5,points)
        if np.any(timeArray < 10):
            break
        try:
            plot_data_r = np.vstack([plot_data_r, intp_r_pred(timeArray) - intp_r_true(timeArray)])
            plot_data_c = np.vstack([plot_data_c, intp_c_pred(timeArray) - intp_c_true(timeArray)])
            if behav_result[i] == 'A':
                plot_data_r_A = np.vstack([plot_data_r_A, intp_r_pred(timeArray) - intp_r_true(timeArray)])
                plot_data_c_A = np.vstack([plot_data_c_A, intp_c_pred(timeArray) - intp_c_true(timeArray)])
            else:
                plot_data_r_E = np.vstack([plot_data_r_E, intp_r_pred(timeArray) - intp_r_true(timeArray)])
                plot_data_c_E = np.vstack([plot_data_c_E, intp_c_pred(timeArray) - intp_c_true(timeArray)])

        except ValueError:
            break

plt.figure(1)
plt.clf()
plt.gcf().set_size_inches((5.61,7.71))
#plt.plot(np.mean(plot_data_r,axis=0), np.linspace(-5, +5, points), color='k')
#plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_r,axis=0)+np.std(plot_data_r,axis=0)/(plot_data_r.shape[0]**0.5), np.mean(plot_data_r,axis=0)-np.std(plot_data_r,axis=0)/(plot_data_r.shape[0]**0.5), color=[0.8, 0.8, 0.8], alpha=0.3)
plt.plot(np.mean(plot_data_r_A,axis=0), np.linspace(-5, +5, points), color='r')
plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_r_A,axis=0)+np.std(plot_data_r_A,axis=0)/(plot_data_r_A.shape[0]**0.5), np.mean(plot_data_r_A,axis=0)-np.std(plot_data_r_A,axis=0)/(plot_data_r_A.shape[0]**0.5), color=[0.8, 0, 0], alpha=0.1)
plt.plot(np.mean(plot_data_r_E,axis=0), np.linspace(-5, +5, points), color='g')
plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_r_E,axis=0)+np.std(plot_data_r_E,axis=0)/(plot_data_r_E.shape[0]**0.5), np.mean(plot_data_r_E,axis=0)-np.std(plot_data_r_E,axis=0)/(plot_data_r_E.shape[0]**0.5), color=[0, 0.8, 0], alpha=0.1)
plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 'r')
plt.vlines(0, -5, +5, 'b')
plt.title('Row (Locked on the First Lick)')
plt.ylabel('Time(s)')
plt.xlabel('Error (Pred-True) (px)')
plt.legend(['Avoid', 'Escape'])
plt.gca().invert_yaxis()


plt.figure(2)
plt.clf()
plt.gcf().set_size_inches((5.61,7.71))
#plt.plot(np.mean(plot_data_c,axis=0), np.linspace(-5, +5, points), color='k')
#plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_c,axis=0)+np.std(plot_data_c,axis=0)/(plot_data_c.shape[0]**0.5), np.mean(plot_data_c,axis=0)-np.std(plot_data_c,axis=0)/(plot_data_c.shape[0]**0.5), color=[0.8, 0.8, 0.8], alpha=0.3)
plt.plot(np.mean(plot_data_c_A,axis=0), np.linspace(-5, +5, points), color='r')
plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_c_A,axis=0)+np.std(plot_data_c_A,axis=0)/(plot_data_c_A.shape[0]**0.5), np.mean(plot_data_c_A,axis=0)-np.std(plot_data_c_A,axis=0)/(plot_data_c_A.shape[0]**0.5), color=[0.8, 0, 0], alpha=0.1)
plt.plot(np.mean(plot_data_c_E,axis=0), np.linspace(-5, +5, points), color='g')
plt.fill_betweenx(np.linspace(-5, +5, points), np.mean(plot_data_c_E,axis=0)+np.std(plot_data_c_E,axis=0)/(plot_data_c_E.shape[0]**0.5), np.mean(plot_data_c_E,axis=0)-np.std(plot_data_c_E,axis=0)/(plot_data_c_E.shape[0]**0.5), color=[0, 0.8, 0], alpha=0.1)
plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 'r')
plt.vlines(0, -5, +5, 'b')
plt.title('Col (Locked on the First Lick)')
plt.ylabel('Time(s)')
plt.xlabel('Error (Pred-True) (px)')
plt.legend(['Avoid', 'Escape'])
plt.gca().invert_yaxis()
