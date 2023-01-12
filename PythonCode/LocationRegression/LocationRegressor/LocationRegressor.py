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
parser.add_argument('--removeEngagedData', default=False, required=False)
args = parser.parse_args()

def NeuralRegressor(tankPath, outputPath, dataset, device, neural_data_rate, truncatedTime_s, train_epoch, init_lr, PFI_numRepeat, numBin, removeNestingData=False, removeEngagedData=False):
    rng = default_rng()
    # Load Tank
    tank_name = re.search('#.*', str(tankPath))[0]
    print(tank_name)

    # Load Data
    neural_data, y_r, y_c, midPointTimes = loadData(tankPath, neural_data_rate, truncatedTime_s, removeNestingData, removeEngagedData)
    print(neural_data.shape)

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
OutputFolder = Path('/home/ubuntu/Data/FineDistanceResult_rmEngaged')
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
            PFI_numRepeat=1, # used 50 in the original code. changed for remove Engaged Data
            numBin=1,
            removeNestingData=args.removeNestingData,
            removeEngagedData=args.removeEngagedData
            )

