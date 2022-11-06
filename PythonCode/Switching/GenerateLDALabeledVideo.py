"""
GeneratePCALabeledVideo
@2022Knowblesse
After PCA dimensionality reduction on the wholesessionunitdata, draw clustered label on the video.
"""

from pathlib import Path
import cv2 as cv
from scipy.interpolate import interp1d
import numpy as np
import re
from tqdm import tqdm
from scipy.io import loadmat
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from  LocationRegression.LocationRegressor.LocationRegressionHelper import loadData
from pathlib import Path
import os
from warnings import warn

tankName = '#20JUN1-200827-171419_PL'
outputPath = Path(r"D:/Data/Lobster")
locationDataPath = Path(r"D:/Data/Lobster/FineDistanceDataset") / Path(tankName)
behaviorDataPath = Path(r"D:/Data/Lobster/BehaviorData") / Path(tankName).with_suffix('.mat')
videoPath = next( (Path(r"D:/Data/Lobster/Lobster_Recording-200319-161008/Data") / Path(tankName)).glob('*.avi') )
videoOutputPath = outputPath / (tankName + '_labeled_LDA.avi')

neural_data, y_r, y_c, midPointTimes = loadData(locationDataPath, neural_data_rate=20, truncatedTime_s=10, removeNestingData=False)
neural_data = np.clip(neural_data, -5, 5)
behavior_data = loadmat(behaviorDataPath)
#'Attacks', 'IRs', 'Licks', 'ParsedData', 'Trials'

# Parse behavior Data
ParsedData = behavior_data['ParsedData']
numTrial = len(ParsedData)
Trials = np.concatenate(ParsedData[:,0])
IRs = np.concatenate(ParsedData[:,1])

AEResult = []
for trial, ir, lick, attack in ParsedData:
    nearAttackIRindex = np.where(ir[:, 0] < attack[0, 0])[0][-1] # the last iron which is earlier than attack
    IAttackIROFI = ir[nearAttackIRindex, 1] - attack[0, 0]
    AEResult.append(int(IAttackIROFI >= 0)) # 0 avoid 1 escape

# Generate Class Vectors

isEncounterZone = np.zeros(midPointTimes.shape, dtype=bool)
isNestingZone = np.zeros(midPointTimes.shape, dtype=bool)
inWhichTrial = np.zeros(midPointTimes.shape)
inAE = np.zeros(midPointTimes.shape)
for i in range(len(midPointTimes)):
    isEncounterZone[i] = np.any((IRs[:, 0] < midPointTimes[i]) & (midPointTimes[i] < IRs[:, 1]))
    isNestingZone[i] = y_c[i] < 200
    idx = np.where(Trials[:,0] > midPointTimes[i])[0] # cf. trial start from 1. 0 trial means activation before the first trial begins
    if len(idx) == 0:
        idx = Trials.shape[0]
    else:
        idx = idx[0]
    inWhichTrial[i] = idx
    if idx == 0:
        inAE[i] = -1
        warn('0 trials exist')
    inAE[i] = AEResult[idx-1]

zoneClass = (~isNestingZone).astype(int) + isEncounterZone.astype(int)
# 0 : nesting
# 1 : foraging
# 2 : encounter


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(neural_data, zoneClass)
neural_data_transformed = lda.transform(neural_data)

c_nesting = (np.mean(neural_data_transformed[zoneClass==0, 0]), np.mean(neural_data_transformed[zoneClass==0, 1]))
c_foraging = (np.mean(neural_data_transformed[zoneClass==1, 0]), np.mean(neural_data_transformed[zoneClass==1, 1]))
c_encounter = (np.mean(neural_data_transformed[zoneClass==2, 0]), np.mean(neural_data_transformed[zoneClass==2, 1]))


def convert2px(point):
    x = point[0]
    y = point[1]
    zeropoint = [250, 250]
    multiplier = [40, 40]
    return (np.round(x*multiplier[0] + zeropoint[0]).astype(int), np.round(y*multiplier[1] + zeropoint[1]).astype(int))



# Prepare LDA interpolation
intp_LDA1 = interp1d(midPointTimes, neural_data_transformed[:,0], kind='linear')
intp_LDA2 = interp1d(midPointTimes, neural_data_transformed[:,1], kind='linear')

## Open Video
vc = cv.VideoCapture(str(videoPath.absolute()))
fps = vc.get(cv.CAP_PROP_FPS)
vw = cv.VideoWriter(str(videoOutputPath),
                    cv.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    (int(vc.get(cv.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))),
                    isColor=True)
_fcount = vc.get(cv.CAP_PROP_FRAME_COUNT)

# Generate Video
frameCount = 0
pca_color_index = 0
ret = True
for i in tqdm(range(int(_fcount))):
    ret, image = vc.read()
    if not ret:
        break
    if (frameCount / fps >= midPointTimes[0]) and (frameCount / fps < midPointTimes[-1]):

        # Draw centers first
        cv.circle(image, convert2px(c_nesting), 5, [255,0,0], -1) # BGR (blue)
        cv.circle(image, convert2px(c_foraging), 5, [0,255,0], -1) # Green
        cv.circle(image, convert2px(c_encounter), 5, [0,0,255], -1) # Red

        # Draw current neural point
        lda1 = np.round(intp_LDA1(frameCount/fps)).astype(int)
        lda2 = np.round(intp_LDA2(frameCount / fps)).astype(int)

        cv.circle(image, convert2px((lda1, lda2)), 5, [0, 255,255], -1) # Yellow

    vw.write(image)

    frameCount += 1

vw.release()
