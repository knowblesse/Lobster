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
import argparse
from torch.utils.data import TensorDataset, DataLoader
import requests

print("Code is running on : " + ("cuda" if torch.cuda.is_available else "cpu"))
time.sleep(1)

parser = argparse.ArgumentParser(prog='LocationRegressor_PFI')
parser.add_argument('regressor')
parser.add_argument('--removeNestingData', default=False, required=False)
args = parser.parse_args()

def loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData=False):
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

    y_r = intp_r(midPointTimes * video_frame_rate)
    y_c = intp_c(midPointTimes * video_frame_rate)

    # If removeNestingData is set True, remove all points which has the column value smaller than 200
    if removeNestingData:
        print('removing nesting')
        neural_data = neural_data[y_c >= 200, :]
        y_r = y_r[y_c >= 200]
        y_c = y_c[y_c >= 200]

    return(neural_data, np.expand_dims(y_r, 1), np.expand_dims(y_c, 1))

def NeuralRegressor(tankPath, outputPath, dataset, device, neural_data_rate, truncatedTime_s, train_epoch, init_lr, PFI_numRepeat, numBin, removeNestingData=False):
    rng = default_rng()
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c = loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData)

    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)
    if dataset == 'distance':
        y = ( (y_r - 280) ** 2 + (y_c - 640) ** 2 ) ** 0.5
    elif dataset == 'row':
        y = y_r
    elif dataset == 'column':
        y = y_c
    else:
        raise(BaseException('Wrong dataset. use distance, row, or column'))

    # Prepare array to store regression result from the test dataset
    WholeTestResult = np.zeros([X.shape[0], 5])  # num data x [row, col, true, fake, predicted]
    WholeTestResult[:, :3] = np.hstack((y_r, y_c, y))

    # Prepare array to store test dataset from the unit shuffled test dataset
    numUnit = int(X.shape[1] / numBin)
    print(f'numUnit : {numUnit}')
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
        optimizer_real = torch.optim.SGD(net_real.parameters(), lr=init_lr, momentum=0.3, weight_decay=0.0)
        optimizer_fake = torch.optim.SGD(net_fake.parameters(), lr=init_lr, momentum=0.3, weight_decay=0.0)
        scheduler_real = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_real, patience=300, cooldown=100)
        scheduler_fake = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fake, patience=300, cooldown=100)
        earlyStopping = EarlyStopping(model=net_real, model_control=net_fake, tolerance=1000, save_best=True)


        # Train
        pbar = tqdm(np.arange(train_epoch))

        for e in pbar:
            # Update net_real
            net_real.train()
            loss_real = F.mse_loss(net_real.forward(X_train), y_train)
            optimizer_real.zero_grad()
            loss_real.backward()
            torch.nn.utils.clip_grad_norm_(net_real.parameters(), 5)
            optimizer_real.step()

            # Update net_fake
            net_fake.train()
            loss_fake = F.mse_loss(net_fake.forward(X_train_shuffled), y_train)
            optimizer_fake.zero_grad()
            loss_fake.backward()
            torch.nn.utils.clip_grad_norm_(net_fake.parameters(), 5)
            optimizer_fake.step()

            # Get learning rate
            lr = [group['lr'] for group in optimizer_real.param_groups]

            # Update tqdm part
            net_real.eval()
            net_fake.eval()
            with torch.no_grad():
                loss_train_real = F.mse_loss(net_real.forward(X_train), y_train)
                loss_train_fake = F.mse_loss(net_fake.forward(X_train_shuffled), y_train)
                loss_test_real = F.mse_loss(net_real.forward(X_test), y_test)
                loss_test_fake = F.mse_loss(net_fake.forward(X_test), y_test)

            pbar.set_postfix_str(\
                    f'lr:{lr[0]:.0e} ' +
                    f'fk:{torch.mean(loss_fake).item():.2f} ' +
                    f'pr:{torch.mean(loss_real).item():.2f} ' +
                    f'fk(Test):{torch.mean(loss_test_fake).item():.2f} ' +
                    f'pr(Test):{torch.mean(loss_test_real).item():.2f} ')
            scheduler_real.step(loss_real)
            scheduler_fake.step(loss_fake)

            # EarlyStopping
            if(earlyStopping(loss_test_real)):
                break

        earlyStopping.loadBest()

        with torch.no_grad():
            loss_test_real = F.mse_loss(net_real.forward(X_test), y_test)
        print(f'Loss : {torch.mean(loss_test_real).item():.2f}')

        # Generate Regression result for test data
        net_real.eval()
        net_fake.eval()
        with torch.no_grad():
            fakeFit = net_fake.forward(X_test)
            realFit = net_real.forward(X_test)


        WholeTestResult[test_index, 3:4] = fakeFit.to('cpu').numpy()
        WholeTestResult[test_index, 4:5] = realFit.to('cpu').numpy()

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

    savemat(outputPath/f'{tank_name}result_{dataset}.mat', 
            {'WholeTestResult': WholeTestResult, 'PFITestResult': PFITestResult})
    
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

InputFolder = Path('/home/ubuntu/Data/FineDistanceDataset')
OutputFolder = Path('/home/ubuntu/Data/FineDistanceResult')
for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    print(f'{i:02} {tank}')
    NeuralRegressor(
            tankPath=tank,
            outputPath=OutputFolder,
            dataset=args.regressor,
            device=device,
            neural_data_rate=20,
            truncatedTime_s=10,
            train_epoch=20000,
            init_lr=0.005,
            PFI_numRepeat=50,
            numBin=1,
            removeNestingData=args.removeNestingData
            )

requests.get(
    'https://api.telegram.org/bot5269105245:AAGCdJAZ9fzfazxC8nc-WI6MTSrxn2QC52U/sendMessage?chat_id=5520161508&text=Done')
