"""
GeneratePCALabeledVideo
@2022Knowblesse
After PCA dimensionality reduction on the wholesessionunitdata, draw clustered label on the video.
"""

import cv2 as cv
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from Switching.SwitchingHelper import parseAllData, getZoneLDA
from pathlib import Path

tankName = '#20JUN1-200827-171419_PL'
videoPath = next( (Path(r"D:/Data/Lobster/Lobster_Recording-200319-161008/Data") / Path(tankName)).glob('*.avi') )
videoOutputPath = Path(r"D:/Data/Lobster") / (tankName + '_labeled_LDA.avi')

data = parseAllData(tankName)
neural_data = data['neural_data']
zoneClass = data['zoneClass']
midPointTimes = data['midPointTimes']

neural_data_transformed, centroids = getZoneLDA(neural_data, zoneClass)

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
        cv.circle(image, convert2px(centroids['nest']), 5, [255,0,0], -1) # BGR (blue)
        cv.circle(image, convert2px(centroids['foraging']), 5, [0,255,0], -1) # Green
        cv.circle(image, convert2px(centroids['encounter']), 5, [0,0,255], -1) # Red

        # Draw current neural point
        lda1 = np.round(intp_LDA1(frameCount/fps)).astype(int)
        lda2 = np.round(intp_LDA2(frameCount / fps)).astype(int)

        cv.circle(image, convert2px((lda1, lda2)), 5, [0, 255,255], -1) # Yellow

    vw.write(image)

    frameCount += 1

vw.release()
