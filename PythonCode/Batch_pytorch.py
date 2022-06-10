"""
Location Decoder Pytorch
@ 2022 Knowblesse
Using deep Neural Net based Regressor, decode the current animal's location using the neural data.
Input :
    TANK_location
The Tank must have these two data
    1) buttered_data.csv (from butter package)
    2) neural spike data (get it from the Matlab)
"""
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
import requests
import csv

class dANN(nn.Module):
    def __init__(self, params):
        # params : device , input_size
        super(dANN, self).__init__()

        self.device = params['device']
        self.fc1 = nn.Linear(
            params['input_size'],
            100)
        self.fc2 = nn.Linear(
            100,
            50)
        self.fc3 = nn.Linear(
            50,
            25)
        self.fc4 = nn.Linear(
            25,
            3)

    def forward(self, x):
        x = x.to(self.device)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.1)


# Constant
neural_data_rate = 2 # datapoints per sec. This shows how many X data is present per second.
truncatedTime_s = 10 # sec. matlab data delete the first and the last 10 sec of the neural data.
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
train_epoch = 10000

FolderLocation = Path(r'/media/ainav/409D-B7E7/21AUG4')
OutputFileLocation = Path(r'/media/ainav/409D-B7E7/Output')

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
    videoFileName = [i for i in TANK_location.glob('*.avi')]
    if len(videoFileName) == 0:
        videoFileName = [i for i in TANK_location.glob('*.mp4')]
    vc = cv.VideoCapture(str(videoFileName[0]))
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
    kf = KFold(n_splits=5, shuffle=True)
    WholeTestResult = np.zeros([X.shape[0], 9])  # num data x [real row, col, deg, fake predicted row, col, deg, predicted row, col, deg]
    WholeTestResult[:, :3] = y

    for train_index, test_index in kf.split(X):

        X_train = torch.tensor(X[train_index,:],dtype=torch.float32, device=device, requires_grad=True)
        X_test = torch.tensor(X[test_index,:], dtype=torch.float32, device=device, requires_grad=False)
        _X_train_shuffled = X[train_index,:].copy()
        np.random.shuffle(_X_train_shuffled)
        X_train_shuffled = torch.tensor(_X_train_shuffled, dtype=torch.float32, device=device, requires_grad=True)

        y_train = torch.tensor(y[train_index,:], dtype=torch.float32, device=device, requires_grad=False)
        y_test = torch.tensor(y[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        params = {'input_size':X_train.shape[1], 'device':device}
        net = dANN(params).to(device)
        net_fake = dANN(params).to(device)
        net.init_weights()
        net_fake.init_weights()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.7)
        optimizer_fake = torch.optim.SGD(net_fake.parameters(), lr=0.0001, momentum=0.7)
        net.train()
        net_fake.train()

        pbar = tqdm(np.arange(train_epoch))

        for e in pbar:
            # Update net
            loss = F.mse_loss(net.forward(X_train), y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

            # Update net_fake
            loss_fake = F.mse_loss(net_fake.forward(X_train_shuffled), y_train)
            optimizer_fake.zero_grad()
            loss_fake.backward()
            torch.nn.utils.clip_grad_norm_(net_fake.parameters(), 5)
            optimizer_fake.step()

            if e % 20 == 0:
                with torch.no_grad():
                    loss = F.mse_loss(net.forward(X_train), y_train, reduction='none') ** 0.5
                    testLoss = F.mse_loss(net.forward(X_test), y_test, reduction='none') ** 0.5
                    testLoss_fake = F.mse_loss(net_fake.forward(X_test), y_test, reduction='none') ** 0.5
                stats = {'epoch': e,
                         'train loss': [round(torch.mean(loss[:,0]).item(),2), round(torch.mean(loss[:,1]).item(),2), round(torch.mean(loss[:,2]).item(),2)],
                         'validation loss fake': [round(torch.mean(testLoss_fake[:, 0]).item(), 2),
                                                  round(torch.mean(testLoss_fake[:, 1]).item(), 2),
                                                  round(torch.mean(testLoss_fake[:, 2]).item(), 2)],
                         'validation loss': [round(torch.mean(testLoss[:, 0]).item(), 2),
                                             round(torch.mean(testLoss[:, 1]).item(), 2),
                                             round(torch.mean(testLoss[:, 2]).item(), 2)]
                         }
                pbar.set_postfix(stats)

        net.eval()
        net_fake.eval()
        with torch.no_grad():
            fakeFit = net_fake.forward(X_test)
            realFit = net.forward(X_test)


        WholeTestResult[test_index,3:6] = fakeFit.to('cpu').numpy()
        WholeTestResult[test_index,6: ] = realFit.to('cpu').numpy()

    print(f"{tank_name} : "
          f"Row : {np.mean((WholeTestResult[:,0]-WholeTestResult[:,3])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,0]-WholeTestResult[:,6])**2)**0.5:.3f},  "
          f"Col : {np.mean((WholeTestResult[:,1]-WholeTestResult[:,4])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,1]-WholeTestResult[:,7])**2)**0.5:.3f},  "
          f"Deg : {np.mean((WholeTestResult[:,2]-WholeTestResult[:,5])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,2]-WholeTestResult[:,8])**2)**0.5:.3f}")
    np.savetxt(str(OutputFileLocation / (tank_name + 'result.csv')),WholeTestResult, fmt='%.3f', delimiter=',')

    with open(str(OutputFileLocation / 'log.csv'), 'a', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow([tank_name,
                     np.mean((WholeTestResult[:,0]-WholeTestResult[:,3])**2)**0.5, np.mean((WholeTestResult[:,0]-WholeTestResult[:,6])**2)**0.5,
                     np.mean((WholeTestResult[:,1]-WholeTestResult[:,4])**2)**0.5, np.mean((WholeTestResult[:,1]-WholeTestResult[:,7])**2)**0.5,
                     np.mean((WholeTestResult[:,2]-WholeTestResult[:,5])**2)**0.5, np.mean((WholeTestResult[:,2]-WholeTestResult[:,8])**2)**0.5])

