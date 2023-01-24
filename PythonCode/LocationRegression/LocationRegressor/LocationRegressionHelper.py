try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    print('pytorch is not installed. Using without it.')
import sys
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.io import loadmat

if 'torch' in sys.modules:
    class dANN(nn.Module):
        def __init__(self, params):
            # params : device , input_size
            super(dANN, self).__init__()
            self.device = params['device']
            self.fc1 = nn.Linear(
                params['input_size'],
                100)
            self.dp1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(
                100,
                50)
            self.dp2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(
                50,
                50)
            self.fc4 = nn.Linear(
                50,
                params['output_node'])


        def forward(self, x):
            x = x.to(self.device)

            x = self.fc1(x)
            x = F.relu(x)

            x = self.dp1(x)

            x = self.fc2(x)
            x = F.relu(x)

            x = self.dp2(x)

            x = self.fc3(x)
            x = torch.tanh(x)

            x = self.fc4(x)

            return x


        def init_weights(self):
            nn.init.normal_(self.fc1.weight, mean=0, std=0.2)
            nn.init.normal_(self.fc2.weight, mean=0, std=0.2)
            nn.init.normal_(self.fc3.weight, mean=0, std=0.2)
            nn.init.normal_(self.fc4.weight, mean=0, std=0.2)


    class EarlyStopping():
        def __init__(self, model, model_control, tolerance=100, save_best=False):
            self.lowest_loss = np.inf
            self.tolerance = tolerance  # how many epochs to stay patient
            self.tolerance_counter = 0
            self.save_best = save_best
            self.model = model
            self.model_control = model_control
            self.model_ever_saved = False


        def __call__(self, loss):
            if loss >= self.lowest_loss:  # worse result
                self.tolerance_counter += 1
                if self.tolerance_counter > self.tolerance:
                    return True
            else:  # better result
                self.lowest_loss = loss
                self.tolerance_counter = 0
                if self.save_best:
                    torch.save(self.model.state_dict(), Path('./tempModel'))
                    torch.save(self.model_control.state_dict(), Path('./tempControlModel'))
                    self.model_ever_saved = True
            return False


        def loadBest(self):
            if not self.save_best:
                raise(BaseException("Earlystopping : 'save_best' was set as False"))
            if not self.model_ever_saved:
                raise(BaseException("Earlystopping : saved model does not exist"))
            self.model.load_state_dict(torch.load(Path('./tempModel')))
            self.model_control.load_state_dict(torch.load(Path('./tempControlModel')))
            self.model.eval()
            self.model_control.eval()


def correctRotationOffset(rotationData):
    # Correct Rotation data for further interpolation.
    # If the degree difference of two consecutive labeled data point is bigger than 180 degree,
    # it is more reasonable to think that the actual rotation is smaller than 180 degree, and
    # crossing the boarder between 0 and 360
    prev_head_direction = rotationData[0]
    degree_offset_value = np.zeros(rotationData.shape[0])
    for i in np.arange(1, rotationData.shape[0]):
        # if the degree change is more than a half rotation, use the smaller rotation value instead.
        if np.abs(rotationData[i] - prev_head_direction) > 180:
            if rotationData[i] > prev_head_direction:
                degree_offset_value[i:] -= 360
            else:
                degree_offset_value[i:] += 360
        prev_head_direction = rotationData[i]
    return rotationData + degree_offset_value

def loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData=False, removeWanderData=False, stratifyData=False):

    default_rng = np.random.default_rng()

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
    # fpsFileName = tankPath / 'FPS.txt'
    # video_frame_rate = int(np.loadtxt(fpsFileName))

    # Check actual timepoint of the frame
    frameInfo_location = [p for p in tankPath.glob('*_frameInfo.mat')]
    if len(frameInfo_location) != 1:
        raise (BaseException("Can not find proper frameInfo.mat"))
    frameInfo = loadmat(str(frameInfo_location[0]))
    frameNumber = np.squeeze(frameInfo['frameNumber'])
    frameTime = np.squeeze(frameInfo['frameTime'])

    # cure weird frame number
    wrongFrameNumberIndex = np.where(np.diff(frameNumber) < 0)[0]
    frameNumber[wrongFrameNumberIndex] = (frameNumber[wrongFrameNumberIndex-1] + frameNumber[wrongFrameNumberIndex+1]) / 2

    intp_frame = interp1d(frameTime, frameNumber, kind='linear')

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

    y_r = intp_r(intp_frame(midPointTimes))
    y_c = intp_c(intp_frame(midPointTimes))

    # If removeNestingData is set True, remove all points which has the column value smaller than 225
    if removeNestingData:
        print('removing nesting')
        neural_data = neural_data[y_c >= 225, :]
        y_r = y_r[y_c >= 225]
        y_c = y_c[y_c >= 225]
        midPointTimes = midPointTimes[y_c >= 225]

    # If removeWanderData is set True, load behavior data and remove all points of following condition
    #   1. the animal is in the nest zone
    #   2. TROF to IRON interval is longer than 12 sec
    if removeWanderData:
        print('removing wander')
        behaviorDataBasePath = Path(r"/home/ubuntu/Data/BehaviorData/")
        behaviorDataPath = behaviorDataBasePath / str(tankPath.name + ".mat")
        behavior_data = loadmat(behaviorDataPath)
        ParsedData = behavior_data['ParsedData']
        numTrial = len(ParsedData)

        # get delete indice
        deleteIndex = np.zeros(midPointTimes.shape, dtype=bool)
        for trial in range(1, numTrial): # skip first trial
            betweenTRON_firstIRON = np.logical_and(
                ParsedData[trial, 0][0,0] <= midPointTimes,
                midPointTimes <  (ParsedData[trial, 1][0,0] + ParsedData[trial, 0][0,0])
            )
            betweenTROF_firstIRON = np.logical_and(
                ParsedData[trial-1, 0][0,1] <= midPointTimes,
                midPointTimes <  (ParsedData[trial, 1][0,0] + ParsedData[trial, 0][0,0])
            )
            latency2HeadEntry = ParsedData[trial, 1][0, 0]
            TROF2IRONInterval = ParsedData[trial, 0][0, 0] - ParsedData[trial-1, 0][0,1] + latency2HeadEntry
            if TROF2IRONInterval <= 12: # Engaged
                continue

            deleteIndex[np.logical_and(
                y_c <= 225, 
                betweenTROF_firstIRON
                )] = True

        neural_data = neural_data[np.logical_not(deleteIndex), :]
        y_r = y_r[np.logical_not(deleteIndex)]
        y_c = y_c[np.logical_not(deleteIndex)]
        midPointTimes = midPointTimes[np.logical_not(deleteIndex)]

    if stratifyData:
        print('stratifying Data')
        behaviorDataBasePath = Path(r"D:\Data\Lobster\BehaviorData")
        behaviorDataPath = behaviorDataBasePath / str(tankPath.name + ".mat")
        behavior_data = loadmat(behaviorDataPath)
        ParsedData = behavior_data['ParsedData']
        numTrial = len(ParsedData)

        # Collect IR Info
        IRs = np.empty((0, 2))
        for trialIdx in range(numTrial):
            IRs = np.vstack((IRs, ParsedData[trialIdx, 1] + ParsedData[trialIdx, 0][0, 0]))

        # Get Zone Info
        isEncounterZone = np.zeros(midPointTimes.shape, dtype=bool)
        isNestingZone = np.zeros(midPointTimes.shape, dtype=bool)
        for i in range(len(midPointTimes)):
            isEncounterZone[i] = np.any((IRs[:, 0] < midPointTimes[i]) & (midPointTimes[i] < IRs[:, 1]))
            isNestingZone[i] = y_c[i] < 225

        zoneClass = (~isNestingZone).astype(int) + isEncounterZone.astype(int)
        datacount = np.bincount(((~isNestingZone).astype(int) + isEncounterZone.astype(int)))
        print(f'N-zone {datacount[0]}, F-zone {datacount[1]}, E-zone {datacount[2]} detected. using {np.min(datacount)} dataset')

        selectedIndex = np.concatenate((default_rng.choice(np.where(zoneClass == 0)[0], np.min(datacount)),
                        default_rng.choice(np.where(zoneClass == 1)[0], np.min(datacount)),
                        default_rng.choice(np.where(zoneClass == 2)[0], np.min(datacount))))

        neural_data = neural_data[selectedIndex,:]
        y_r = y_r[selectedIndex]
        y_c = y_c[selectedIndex]
        midPointTimes = midPointTimes[selectedIndex]


    return(neural_data, np.expand_dims(y_r, 1), np.expand_dims(y_c, 1), midPointTimes)

# tankPath = Path('D:\Data\Lobster\FineDistanceDataset\#20JUN1-200831-110125_PL')
# neural_data, y_r, y_c, midPointTimes = loadData(tankPath, neural_data_rate=20, truncatedTime_s=10, removeEnagedData=False, removeNestingData=False)
