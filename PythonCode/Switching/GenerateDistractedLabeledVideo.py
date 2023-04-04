"""
GeneratePCALabeledVideo
@2022Knowblesse
After PCA dimensionality reduction on the wholesessionunitdata, draw clustered label on the video.
"""

import cv2 as cv
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat
from Switching.SwitchingHelper import parseAllData
from pathlib import Path
TANK_NAME = "#21JAN5-210723-180401_IL"
TANK_PATH = Path(r"D:/Data/Lobster/Lobster_Recording-200319-161008/Data") / TANK_NAME
videoPath = next( TANK_PATH.glob('*.avi') )
videoOutputPath = Path.home() / ('Desktop/' +  TANK_NAME + '_labeled_.avi')

# Find distracted label data
if [path for path in Path(TANK_PATH).glob('bool_distracted.csv')]:
    data_distracted = np.loadtxt(next(TANK_PATH.glob('bool_distracted.csv')), delimiter=',')
else:
    raise(BaseException("Can not find distracted label data"))

# Find WholeTestResult (Fine Location Regression Data)
REGR_PATH = Path("D:\Data\Lobster\FineDistanceResult_syncFixed")
if [path for path in REGR_PATH.glob(TANK_NAME + '*')]:
    data_regr = loadmat(next(REGR_PATH.glob(TANK_NAME + '*')))
    data_regr = data_regr['WholeTestResult']
L1Errors = (data_regr[:,2] - data_regr[:,4]) * 0.169

# moving average
L1Errors = np.convolve(L1Errors, np.ones(20), 'same') / 20

# Get midPointTimes
data = parseAllData(TANK_NAME)
midPointTimes = data['midPointTimes']

# Get Frame time
data_frame = loadmat(next((Path(r'D:\Data\Lobster\FineDistanceDataset') / TANK_NAME).glob('*frameInfo.mat')))
frameNumber = np.squeeze(data_frame['frameNumber'])
frameTime = np.squeeze(data_frame['frameTime'])

# cure weird frame number
wrongFrameNumberIndex = np.where(np.diff(frameNumber) < 0)[0]
frameNumber[wrongFrameNumberIndex] = (frameNumber[wrongFrameNumberIndex - 1] + frameNumber[
    wrongFrameNumberIndex + 1]) / 2

# Create (frame to time interp1d object
intp_frame = interp1d(frameNumber, frameTime, kind='linear', bounds_error=False, fill_value=(0, np.inf))

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
    if (intp_frame(frameCount)  >= midPointTimes[0]) and (intp_frame(frameCount) < midPointTimes[-1]):

        # Draw Distracted
        if data_distracted[frameCount] == 1:
            cv.circle(image, (50,100), 25, [0,0,255], -1) # BGR (Red)
            cv.putText(image, f'Distracted', [240, 250],
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=[255, 255, 255], thickness=2)

        # Find midpointtime index
        idx_mpt = np.where(midPointTimes < intp_frame(frameCount))[0][-1]

        # Draw Graph
        L1Error = L1Errors[idx_mpt]
        errorMultiplier = 3
        cv.rectangle(image, [0,380], [20, 380 - int(errorMultiplier * np.abs(L1Error))], [255, 255, 255], -1)
        cv.putText(image, f'{np.abs(L1Error):5.2f}cm', [0, 460],
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=[255, 255, 255], thickness=2)

        # Draw Error circle around the rat
        cv.circle(image, (int(data_regr[idx_mpt, 1]), int(data_regr[idx_mpt,0])), int(np.abs(L1Error)/0.169/2), [255, 255, 255], 2)
        cv.arrowedLine(image, (int(data_regr[idx_mpt, 1]), int(data_regr[idx_mpt,0])), (int(data_regr[idx_mpt, 1] + L1Error/0.169/2) , int(data_regr[idx_mpt,0])), [255, 255, 255], 2)

    vw.write(image)

    frameCount += 1

vw.release()
