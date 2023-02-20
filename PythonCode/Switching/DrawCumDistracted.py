import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.interpolate import interp1d
plt.rcParams["font.family"] = "Noto Sans"
plt.rcParams["font.size"] = 6.6
# Find prev dataset if exist
TANK_PARENT_PATH = Path('D:\Data\Lobster\Lobster_Recording-200319-161008\Data')
sessionList = [path for path in TANK_PARENT_PATH.rglob('*bool*')]

# Load Behavior Data and frame data to find the start point of the experiment
# Load Behavior data
BEHAV_PARENT_PATH = Path(r"D:/Data/Lobster/BehaviorData")

# Get Frame data
FRAME_PARENT_PATH = Path(r'D:\Data\Lobster\FineDistanceDataset')

data = []
max_data_length = 55164
min_data_length = 41863
data_same_length = []
data_clipped = []

fig1 = plt.figure(figsize=(3., 2.7))
ax1 = fig1.subplots(1)
ax1.cla()

for i, path in enumerate(sessionList):
    tank_name = sessionList[i].parent.stem

    # Find Trial1's TRON
    behavior_data = loadmat(BEHAV_PARENT_PATH / (tank_name + '.mat'))
    start_time = behavior_data['ParsedData'][0,0][0,0]

    # Load Frame Data
    data_frame = loadmat(str(next((FRAME_PARENT_PATH / tank_name).glob('*frameInfo.mat'))))
    frameNumber = np.squeeze(data_frame['frameNumber'])
    frameTime = np.squeeze(data_frame['frameTime'])

    # Cure weird frame number
    wrongFrameNumberIndex = np.where(np.diff(frameNumber) < 0)[0]
    frameNumber[wrongFrameNumberIndex] = (frameNumber[wrongFrameNumberIndex - 1] + frameNumber[
        wrongFrameNumberIndex + 1]) / 2

    # Create (frame to time interp1d object
    intp_time2frame = interp1d(frameTime, frameNumber, kind='linear', bounds_error=False, fill_value=(0, np.inf))

    # Load Distracted Data and generate intp object
    cum_distract_data = np.cumsum(np.loadtxt(path, delimiter=','))

    intp_cum_distract = interp1d(np.arange(cum_distract_data.shape[0]), cum_distract_data, kind='linear')

    distract_data = intp_cum_distract(intp_time2frame(np.arange(start_time, start_time + 25*60, 1)))
    distract_data = (distract_data - distract_data[0]) / np.max(distract_data)

    # Process
    data.append(distract_data)
    ax1.plot(distract_data, linestyle=':', color=[0.5, 0.5, 0.5], linewidth=0.5)



ax1.plot(np.mean(np.array(data), 0), color='k', linewidth=2)
ax1.plot(np.arange(data[0].shape[0]), np.linspace(0, 1, data[0].shape[0]), color=[1, 0, 0], linewidth=2, linestyle=':')
ax1.set_xticks(np.arange(0, 60*30, 60*5))
ax1.set_xticklabels(np.arange(0, 30, 5))
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Cumulative Distracted Behavior in N-zone (ratio)')

