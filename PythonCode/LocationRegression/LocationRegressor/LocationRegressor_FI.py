import re
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from LocationRegressionHelper import *
import time
from numpy.random import default_rng
import argparse
import platform
import numpy as np
from pathlib import Path

import requests


print("Code is running on : " + ("cuda" if torch.cuda.is_available else "cpu"))
time.sleep(1)

parser = argparse.ArgumentParser(prog='LocationRegressor_PFI')
parser.add_argument('regressor')
parser.add_argument('--removeNestingData', default='False', required=False)
parser.add_argument('--removeEncounterData', default='False', required=False)
parser.add_argument('--removeWanderData', default='False', required=False)
parser.add_argument('--stratifyData', default='False', required=False)
args = parser.parse_args()

def strinput2bool(str_input):
    if str_input in ('True', 'true', 'y', 'yes'):
        return True
    elif str_input in ('False', 'false', 'n', 'no'):
        return False
    else:
        raise BaseException('Wrong argument input')

# Convert to bool
args.removeNestingData = strinput2bool(args.removeNestingData)
args.removeEncounterData = strinput2bool(args.removeEncounterData)
args.removeWanderData = strinput2bool(args.removeWanderData)
args.stratifyData = strinput2bool(args.stratifyData)

def NeuralRegressor(tankPath, outputPath, dataset, device, neural_data_rate, truncatedTime_s, train_epoch, init_lr, PFI_numRepeat, numBin, removeNestingData, removeEncounterData, removeWanderData, stratifyData):
    rng = default_rng()
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c, y_deg, midPointTimes, zoneClass = loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData, removeEncounterData, removeWanderData, stratifyData)
    print(neural_data.shape)

    # Dataset Prepared
    X = np.clip(neural_data, -5, 5)
    if dataset == 'distance':
        print('Distance Regressor')
        y = ( (y_r - 280) ** 2 + (y_c - 640) ** 2 ) ** 0.5
    elif dataset == 'row':
        print('Row Regressor')
        y = y_r
    elif dataset == 'column':
        print('Column Regressor')
        y = y_c
    elif dataset == 'speed':
        print('Speed Regressor')
        X = X[1:, :]
        y = ( np.diff(y_r,1,0) ** 2 + np.diff(y_c,1,0) ** 2 ) ** 0.5
        y_r = y_r[1:]
        y_c = y_c[1:]
    elif dataset == 'degree':
        print('Degree Regressor')
        y = y_deg
    elif dataset == 'time':
        print('Time Regressor')
        y = np.expand_dims(midPointTimes, 1)
    elif dataset == 'weber':
        print("Distance in Weber's law")
        distance  = ( (y_r - 280) ** 2 + (y_c - 640) ** 2 ) ** 0.5 
        y = 1 / distance * 100
        print(y)
    else:
        raise(BaseException('Wrong dataset. use distance, row, or column'))

    # use well-distributed y values
    # bin y into 5 bins, label each datapoint with the bin, and use the bin label to equally distribute dataset during train/test split.
    useEqualyBin = False
    if useEqualyBin:
        zoneClass = np.floor(np.argsort(np.squeeze(y)) / y.shape[0] * 5).astype(int)

        datacount = np.bincount(zoneClass)

        zoneClass = np.digitize(y, (np.max(y) - np.min(y)) * np.array([0.2, 0.4, 0.6, 0.8]) + np.min(y))

        print(f'zoneCount 0 : {np.sum(zoneClass == 0)} 1 : {np.sum(zoneClass == 1)} 2 : {np.sum(zoneClass == 2)} 3 : {np.sum(zoneClass == 3)} 4 : {np.sum(zoneClass == 4)}')

        datacount = np.bincount(np.squeeze(zoneClass))
        selectedIndex = np.concatenate((rng.choice(np.where(zoneClass == 0)[0], np.min(datacount)),
                        rng.choice(np.where(zoneClass == 1)[0], np.min(datacount)),
                        rng.choice(np.where(zoneClass == 2)[0], np.min(datacount)),
                        rng.choice(np.where(zoneClass == 3)[0], np.min(datacount)),
                        rng.choice(np.where(zoneClass == 4)[0], np.min(datacount))))

        X = X[selectedIndex,:]
        y_r = y_r[selectedIndex]
        y_c = y_c[selectedIndex]
        y = y[selectedIndex]
        midPointTimes = midPointTimes[selectedIndex]
        zoneClass = np.squeeze(zoneClass[selectedIndex])

        print(f'zoneCount(after) 0 : {np.sum(zoneClass == 0)} 1 : {np.sum(zoneClass == 1)} 2 : {np.sum(zoneClass == 2)} 3 : {np.sum(zoneClass == 3)} 4 : {np.sum(zoneClass == 4)}')

    # Prepare array to store regression result from the test dataset
    WholeTestResult = np.zeros([X.shape[0], 5])  # num data x [row, col, true, fake, predicted]
    WholeTestResult[:, :3] = np.hstack((y_r, y_c, y))

    # Prepare array to store test dataset from the unit shuffled test dataset
    numUnit = int(X.shape[1] / numBin)
    print(f'numUnit : {numUnit}')

    # Setup KFold
    CV_split = 5
    kf = StratifiedKFold(n_splits=CV_split, shuffle=True, random_state=622)
    train_log = np.zeros((CV_split, train_epoch, 4)) # loss train_fake train_real test_fake test_real
    current_cv = 0

    """
    For Feature Importance code, 
    read `FI_rank.csv` and select the top 20% of neurons that are most important.
    build two model => _noace and _control
    
    """
    # read FI_rank.csv
    FI_rank = np.loadtxt(tankPath / 'FI_rank.csv', delimiter=',', dtype=int) - 1  # 0-based index

    num2remove = int(np.round(numUnit * 0.2))

    # Create data without ace
    X_noace = X.copy()
    X_noace = X_noace[:, np.sort(FI_rank[:-num2remove])]

    # Create random data
    random_remove = np.random.choice(FI_rank[:-num2remove], num2remove, replace=False) # choose random neurons to remove (not the most important neurons)

    X_control = X.copy()
    X_control = X_control[:, np.delete(np.arange(numUnit), random_remove)]

    # Start training
    for train_index, test_index in kf.split(X, zoneClass):
        X_noace_train = torch.tensor(X_noace[train_index, :], dtype=torch.float32, device=device, requires_grad=True)
        X_noace_test = torch.tensor(X_noace[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        X_control_train = torch.tensor(X_control[train_index, :], dtype=torch.float32, device=device, requires_grad=True)
        X_control_test = torch.tensor(X_control[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        y_train = torch.tensor(y[train_index, :], dtype=torch.float32, device=device, requires_grad=False)
        y_test = torch.tensor(y[test_index, :], dtype=torch.float32, device=device, requires_grad=False)

        params = {'input_size': X_noace_train.shape[1], 'device': device, 'output_node': 1}
        net_noace = dANN(params).to(device)
        net_control = dANN(params).to(device)
        net_noace.init_weights()
        net_control.init_weights()
        optimizer_noace = torch.optim.SGD(net_noace.parameters(), lr=init_lr, momentum=0.3, weight_decay=0.0)
        optimizer_control = torch.optim.SGD(net_control.parameters(), lr=init_lr, momentum=0.3, weight_decay=0.0)
        scheduler_noace = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_noace, patience=300, cooldown=100)
        scheduler_control = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_control, patience=300, cooldown=100)
        earlyStopping = EarlyStopping(model=net_noace, model_control=net_control, tolerance=1000, save_best=True)


        # Train
        pbar = tqdm(np.arange(train_epoch))

        for e in pbar:
            # Update net_noace
            net_noace.train()
            loss_noace = F.mse_loss(net_noace.forward(X_noace_train), y_train)
            optimizer_noace.zero_grad()
            loss_noace.backward()
            torch.nn.utils.clip_grad_norm_(net_noace.parameters(), 5)
            optimizer_noace.step()

            # Update net_control
            net_control.train()
            loss_control = F.mse_loss(net_control.forward(X_control_train), y_train)
            optimizer_control.zero_grad()
            loss_control.backward()
            torch.nn.utils.clip_grad_norm_(net_control.parameters(), 5)
            optimizer_control.step()

            # Get learning rate
            lr = [group['lr'] for group in optimizer_noace.param_groups]

            # Update tqdm part
            net_noace.eval()
            net_control.eval()
            with torch.no_grad():
                loss_train_noace = F.mse_loss(net_noace.forward(X_noace_train), y_train)
                loss_train_control = F.mse_loss(net_control.forward(X_control_train), y_train)
                loss_test_noace = F.mse_loss(net_noace.forward(X_noace_test), y_test)
                loss_test_control = F.mse_loss(net_control.forward(X_control_test), y_test)
                train_log[current_cv, e, :] = np.array([
                    loss_train_control.to('cpu'),
                    loss_train_noace.to('cpu'),
                    loss_test_control.to('cpu'),
                    loss_test_noace.to('cpu')])


            pbar.set_postfix_str(\
                    f'lr:{lr[0]:.0e} ' +
                    f'ctl:{torch.mean(loss_control).item():.2f} ' +
                    f'ace:{torch.mean(loss_noace).item():.2f} ' +
                    f'ctl(Test):{torch.mean(loss_test_control).item():.2f} ' +
                    f'ace(Test):{torch.mean(loss_test_noace).item():.2f} ')
            scheduler_noace.step(loss_noace)
            scheduler_control.step(loss_control)

            # EarlyStopping
            if(earlyStopping(loss_test_noace)):
                break

        earlyStopping.loadBest()

        # Generate Regression result for test data
        net_noace.eval()
        net_control.eval()
        with torch.no_grad():
            controlFit = net_control.forward(X_control_test)
            noaceFit = net_noace.forward(X_noace_test)


        WholeTestResult[test_index, 3:4] = controlFit.to('cpu').numpy()
        WholeTestResult[test_index, 4:5] = noaceFit.to('cpu').numpy()

        # start new CV
        current_cv += 1

    savemat(outputPath/f'{tank_name}result_{dataset}.mat', {
            'WholeTestResult': WholeTestResult, 
            'midPointTimes': midPointTimes,
            'train_log': train_log})
    
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

if platform.system() == 'Windows':
    BasePath = Path('D:\Data\Lobster')
else:
    BasePath = Path.home() / 'Data'


InputFolder = BasePath / 'FineDistanceDataset'
OutputFolder = BasePath / 'FineDistanceResult_syncFixed_240428'

for i, tank in enumerate(sorted([p for p in InputFolder.glob('#*')])):
    print(f'{i:02} {tank}')
    if args.removeWanderData:
        # check if the session has the nnb data
        NNBFolder = BasePath / 'NonNavigationalBehaviorData' 
        if not ((NNBFolder / (tank.name + '_nnb.csv')).is_file()):
            continue
    NeuralRegressor(
            tankPath=tank,
            outputPath=OutputFolder,
            dataset=args.regressor,
            device=device,
            neural_data_rate=20,
            truncatedTime_s=10,
            train_epoch=20000,
            init_lr=0.005,
            PFI_numRepeat=1, # used 50 in the original code. changed for remove Engaged Data
            numBin=1,
            removeNestingData=args.removeNestingData,
            removeEncounterData=args.removeEncounterData,
            removeWanderData=args.removeWanderData,
            stratifyData=args.stratifyData
            )

