"""
GeneratePredictionVideo
@ 2022 Knowblesse
Using the regression data (predicted location), generate a new video with predicted animal location labelled.
Input : 
    TANK_location 
    RegressionResult Path
"""
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import cv2 as cv
from tqdm import tqdm

Tank_Path = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1\#20JUN1-200827-171419_PL')
Video_Path = next(Tank_Path.glob('*.avi'))
VideoOutput_Path = Path(r'F:\Output')
Session_Name = re.search('#.*', str(Tank_Path))[0]
RegressionResult_Path = next(Path(r'F:\Output').glob(Session_Name + '*'))
Truncated_time = 10  # sec
Neural_data_rate = 2  # datapoints per sec. This shows how many X data is present per second.

# Video Setup

vc = cv.VideoCapture(str(Video_Path.absolute()))
fps = vc.get(cv.CAP_PROP_FPS)
vw = cv.VideoWriter(str(VideoOutput_Path / (Session_Name + '_labeled_predicted.avi')),
                    cv.VideoWriter_fourcc(*'DIVX'),
                    fps,
                    (int(vc.get(cv.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))),
                    isColor=True)
_fcount = vc.get(cv.CAP_PROP_FRAME_COUNT)
# RegressionResult Setup
reg_dat = np.loadtxt(str(RegressionResult_Path), delimiter=',')

intp_r_true = interp1d(np.array(np.arange(
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps,
    # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps + reg_dat.shape[0] / Neural_data_rate * fps,
    1 / Neural_data_rate * fps)),
    reg_dat[:, 0],
    kind='linear')
intp_c_true = interp1d(np.array(np.arange(
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps,
    # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
    (Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate) * fps,
    1 / Neural_data_rate * fps)),
    reg_dat[:, 1],
    kind='linear')
intp_r_pred = interp1d(np.array(np.arange(
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps,
    # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps + reg_dat.shape[0] / Neural_data_rate * fps,
    1 / Neural_data_rate * fps)),
    reg_dat[:, 6],
    kind='linear')
intp_c_pred = interp1d(np.array(np.arange(
    (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps,
    # Start from Truncated_time + 0.5*1/Neural_data_rate. Adding "0.5*1/Neural_data_rate" makes the position of location dataset to go in to the middle
    (Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate) * fps,
    1 / Neural_data_rate * fps)),
    reg_dat[:, 7],
    kind='linear')

# Generate Video
frameCount = 0
ret = True
for i in tqdm(range(int(_fcount))):
    ret, image = vc.read()
    if not ret:
        break
    if frameCount >= (Truncated_time + (0.5 * 1 / Neural_data_rate)) * fps:
        if frameCount < (Truncated_time + (0.5 * 1 / Neural_data_rate) + reg_dat.shape[0] / Neural_data_rate - 1 / Neural_data_rate) * fps:
            r_true = np.round(intp_r_true(frameCount)).astype(int)
            c_true = np.round(intp_c_true(frameCount)).astype(int)
            cv.circle(image, (c_true, r_true), 3, [0, 0, 255], -1)
            r_pred = np.round(intp_r_pred(frameCount)).astype(int)
            c_pred = np.round(intp_c_pred(frameCount)).astype(int)
            cv.circle(image, (c_pred, r_pred), 3, [0, 255, 0], -1)

    vw.write(image)

    frameCount += 1

vw.release()
