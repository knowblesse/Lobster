import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.io import savemat
from sklearn.model_selection import KFold
from tqdm import tqdm
import platform
import csv
from LocationRegressionHelper import *
import time
from numpy.random import default_rng

print("Code is running on : " + ("cuda" if torch.cuda.is_available else "cpu"))
time.sleep(1)

def loadData(tankPath, neural_data_rate, truncatedTime_s):
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

    y_r = np.expand_dims(intp_r(midPointTimes * video_frame_rate), 1)
    y_c = np.expand_dims(intp_c(midPointTimes * video_frame_rate), 1)

    return(neural_data, y_r, y_c)

def DistanceRegressor(tankPath, outputPath, device, neural_data_rate, truncatedTime_s, train_epoch, init_lr, PFI_numRepeat, numBin):
    rng = default_rng()
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c = loadData(tankPath, neural_data_rate, truncatedTime_s)

    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)
    y = ( (y_r - 280) ** 2 + (y_c - 640) ** 2 ) ** 0.5

    # Prepare array to store regression result from the test dataset
    WholeTestResult = np.zeros([X.shape[0], 3])  # num data x [row, col, real, fake, predicted]
    WholeTestResult[:, :3] = np.hstack((y_r, y_c, y))

    # Prepare array to store test dataset from the unit shuffled test dataset
    numUnit = int(X.shape[1] / numBin)
    PFITestResult = np.zeros([X.shape[0], numUnit, PFI_numRepeat])
   
    # Setup KFold
    kf = KFold(n_splits=5, shuffle=True)

    # Start training
    for train_index, test_index in kf.split(X):

        X_train = torch.tensor(X[train_index, :], dtype=torch.float32, device=device, requires_grad=True)
        X_test = torch.tensor(X[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        _X_train_shuffled = X[train_index, :].copy()
        np.random.shuffle(_X_train_shuffled)
        X_train_shuffled = torch.tensor(_X_train_shuffled, dtype=torch.float32, device=device, requires_grad=True)

        y_train = torch.tensor(y[train_index, :], dtype=torch.float32, device=device, requires_grad=False)
        y_test = torch.tensor(y[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        params = {'input_size': X_train.shape[1], 'device': device, 'output_node': 1}
        net_real = dANN(params).to(device)
        net_fake = dANN(params).to(device)
        net_real.init_weights()
        net_fake.init_weights()
        optimizer_real = torch.optim.SGD(net_real.parameters(), lr=init_lr, momentum=0.7)
        optimizer_fake = torch.optim.SGD(net_fake.parameters(), lr=init_lr, momentum=0.7)

        # Train
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

        # Generate Regression result for test data
        net_real.eval()
        net_fake.eval()
        with torch.no_grad():
            fakeFit = net_fake.forward(X_test)
            realFit = net_real.forward(X_test)


        WholeTestResult[test_index, 1:2] = fakeFit.to('cpu').numpy()
        WholeTestResult[test_index, 2:3] = realFit.to('cpu').numpy()

        # Generate Regression result for corrupted test data (PFI)
        for unit in range(numUnit):
            for rep in range(PFI_numRepeat):
                X_corrupted = X.copy()
                # shuffle only selected unit data(=corrupted)
                for bin in range(numBin):
                    rng.shuffle(X_corrupted[:, numBin*unit + bin])
                # evaluate corrupted data
                with torch.no_grad():
                    X_test_corrupted = torch.tensor(X_corrupted[test_index, :], dtype=torch.float32, device=device, requires_grad=False)
                    y_corrupted = net_real.forward(X_test_corrupted)
                PFITestResult[test_index, unit, rep] = np.squeeze(y_corrupted.to('cpu').numpy())

    savemat(outputPath/f'{tank_name}result.mat', 
            {'WholeTestResult': WholeTestResult, 'PFITestResult': PFITestResult})
    

# LocationRegression
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
InputFolder = Path('/home/ubuntu/Data/LocationRegressionData')
OutputFolder = Path('/home/ubuntu/Data/DistanceRegressionResult')
for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    print(f'{i:02} {tank}')
    DistanceRegressor(
            tankPath=tank,
            outputPath=OutputFolder, 
            device=device, 
            neural_data_rate=2, 
            truncatedTime_s=10, 
            train_epoch=10000, 
            init_lr=0.0001,
            PFI_numRepeat=30,
            numBin=10
            )
    




