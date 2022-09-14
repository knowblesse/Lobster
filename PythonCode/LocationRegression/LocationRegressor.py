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
import csv
from LocationRegressionHelper import *


# Location Regressor Function
def LocationRegressor(tankPath, outputPath, neural_data_rate, truncatedTime_s, train_epoch, init_lr):
    #Load Tank
    tank_name = re.search('#.*',str(tankPath))[0]
    print(tank_name)

    # Check if the video file is buttered
    butter_location = [p for p in tankPath.glob('*_buttered.csv')]

    if len(butter_location) == 0:
        raise(BaseException("Can not find a butter file in the current Tank location"))
    elif len(butter_location) > 1:
        raise(BaseException("There are multiple files ending with _buttered.csv"))

    # Check if the neural data file is present
    wholeSessionUnitData_location = [p for p in tankPath.glob('*_wholeSessionUnitData.csv')]

    if len(wholeSessionUnitData_location) == 0:
        raise(BaseException("Can not find a regression data file in the current Tank location"))
    elif len(wholeSessionUnitData_location) > 1:
        raise(BaseException("There are multiple files ending with _wholeSessionUnitData.csv"))

    # Check Video FPS
    fpsFileName = tankPath / 'FPS.txt'
    video_frame_rate = int(np.loadtxt(fpsFileName))

    # Load file
    butter_data = np.loadtxt(str(butter_location[0]), delimiter='\t')
    neural_data = np.loadtxt(str(wholeSessionUnitData_location[0]), delimiter=',')

    # Check if -1 value exist in the butter data
    if np.any(butter_data == -1):
        raise(BaseException("-1 exist in the butter data. check with the relabeler"))

    # Generate Interpolation function
    intp_r = interp1d(butter_data[:, 0], butter_data[:, 1], kind='linear')
    intp_c = interp1d(butter_data[:, 0], butter_data[:, 2], kind='linear')
    intp_d = interp1d(butter_data[:, 0], correctRotationOffset(butter_data[:, 3]), kind='linear')

    # Find midpoint of each neural data
    #   > If neural data is collected from 0 ~ 0.5 sec, (neural_data_rate=2), then the mid-point of the
    #       neural data is 0.25 sec. The next neural data, which is collected during 0.5~1.0 sec, has
    #       the mid-point of 0.75 sec.
    midPointTimes = truncatedTime_s + (1/neural_data_rate)*np.arange(neural_data.shape[0]) + 0.5 * (1/neural_data_rate)

    y_r = np.expand_dims(intp_r(midPointTimes * video_frame_rate), 1)
    y_c = np.expand_dims(intp_c(midPointTimes * video_frame_rate), 1)
    y_d = np.expand_dims(intp_d(midPointTimes * video_frame_rate) % 360, 1)
    
    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)
    y = np.concatenate((y_r, y_c, y_d), axis=1)

    # Run Test
    kf = KFold(n_splits=5, shuffle=True)
    WholeTestResult = np.zeros([X.shape[0], 9])  # num data x [real row, col, deg, fake predicted row, col, deg, predicted row, col, deg]
    WholeTestResult[:, :3] = y

    # Start training
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    for train_index, test_index in kf.split(X):

        X_train = torch.tensor(X[train_index,:],dtype=torch.float32, device=device, requires_grad=True)
        X_test = torch.tensor(X[test_index,:], dtype=torch.float32, device=device, requires_grad=False)

        _X_train_shuffled = X[train_index,:].copy()
        np.random.shuffle(_X_train_shuffled)
        X_train_shuffled = torch.tensor(_X_train_shuffled, dtype=torch.float32, device=device, requires_grad=True)

        y_train = torch.tensor(y[train_index,:], dtype=torch.float32, device=device, requires_grad=False)
        y_test = torch.tensor(y[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        params = {'input_size':X_train.shape[1], 'device':device}
        net_real = dANN(params).to(device)
        net_fake = dANN(params).to(device)
        net_real.init_weights()
        net_fake.init_weights()
        optimizer_real = torch.optim.SGD(net_real.parameters(), lr=init_lr, momentum=0.7)
        optimizer_fake = torch.optim.SGD(net_fake.parameters(), lr=init_lr, momentum=0.7)

        pbar = tqdm(np.arange(train_epoch))
        lr = init_lr

        for e in pbar:
            
            # Update Learning rate (moving learning rate)
            if e > 10000:
                for g in optimizer_real.param_groups:
                    lr = g['lr'] * np.exp(-0.0005)
                    g['lr'] = lr
                for g in optimizer_fake.param_groups:
                    lr = g['lr'] * np.exp(-0.0005)
                    g['lr'] = lr
            else:
                lr = init_lr

            # Update net_real
            net_real.train()
            loss = F.mse_loss(net_real.forward(X_train), y_train)
            optimizer_real.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_real.parameters(), 5)
            optimizer_real.step()

            # Update net_fake
            net_fake.train()
            loss_fake = F.mse_loss(net_fake.forward(X_train_shuffled), y_train)
            optimizer_fake.zero_grad()
            loss_fake.backward()
            torch.nn.utils.clip_grad_norm_(net_fake.parameters(), 5)
            optimizer_fake.step()

            # Print Loss
            if e % 20 == 0:
                with torch.no_grad():
                    loss = F.mse_loss(net_real.forward(X_train), y_train, reduction='none') ** 0.5
                    testLoss = F.mse_loss(net_real.forward(X_test), y_test, reduction='none') ** 0.5
                    testLoss_fake = F.mse_loss(net_fake.forward(X_test), y_test, reduction='none') ** 0.5
                stats = {'epoch': e,
                         'lr': lr,
                         'train loss': [round(torch.mean(loss[:,0]).item(),2), round(torch.mean(loss[:,1]).item(),2), round(torch.mean(loss[:,2]).item(),2)],
                         'validation loss fake': [round(torch.mean(testLoss_fake[:, 0]).item(), 2),
                                                  round(torch.mean(testLoss_fake[:, 1]).item(), 2),
                                                  round(torch.mean(testLoss_fake[:, 2]).item(), 2)],
                         'validation loss': [round(torch.mean(testLoss[:, 0]).item(), 2),
                                             round(torch.mean(testLoss[:, 1]).item(), 2),
                                             round(torch.mean(testLoss[:, 2]).item(), 2)]
                         }
                pbar.set_postfix(stats)

        net_real.eval()
        net_fake.eval()
        with torch.no_grad():
            fakeFit = net_fake.forward(X_test)
            realFit = net_real.forward(X_test)

        WholeTestResult[test_index,3:6] = fakeFit.to('cpu').numpy()
        WholeTestResult[test_index,6: ] = realFit.to('cpu').numpy()

    print(f"{tank_name} : "
          f"Row : {np.mean((WholeTestResult[:,0]-WholeTestResult[:,3])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,0]-WholeTestResult[:,6])**2)**0.5:.3f},  "
          f"Col : {np.mean((WholeTestResult[:,1]-WholeTestResult[:,4])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,1]-WholeTestResult[:,7])**2)**0.5:.3f},  "
          f"Deg : {np.mean((WholeTestResult[:,2]-WholeTestResult[:,5])**2)**0.5:.3f} | {np.mean((WholeTestResult[:,2]-WholeTestResult[:,8])**2)**0.5:.3f}")
    np.savetxt(str(outputPath / (tank_name + 'result.csv')),WholeTestResult, fmt='%.3f', delimiter=',')

    with open(str(outputPath / 'log.csv'), 'a', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow([tank_name,
                     np.mean((WholeTestResult[:,0]-WholeTestResult[:,3])**2)**0.5, np.mean((WholeTestResult[:,0]-WholeTestResult[:,6])**2)**0.5,
                     np.mean((WholeTestResult[:,1]-WholeTestResult[:,4])**2)**0.5, np.mean((WholeTestResult[:,1]-WholeTestResult[:,7])**2)**0.5,
                     np.mean((WholeTestResult[:,2]-WholeTestResult[:,5])**2)**0.5, np.mean((WholeTestResult[:,2]-WholeTestResult[:,8])**2)**0.5])


InputFolder =  Path('/home/ainav/Data/LocationRegressionData')
OutputFolder = Path('/home/ainav/Data/LocationRegressionResult')

for tank in InputFolder.glob('#*'):
    print(tank)
    LocationRegressor(tank, OutputFolder, neural_data_rate=2, truncatedTime_s=10, train_epoch=20000, init_lr=0.0001)
